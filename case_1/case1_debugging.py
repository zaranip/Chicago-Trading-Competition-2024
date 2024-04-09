from xchangelib import xchange_client
from xchangelib.xchange_client import Side
import asyncio
import time
import sys
import os

def log_to_file(log_file_path):
    """
    Redirects the console output to a log file.
    Prints to both the console and the log file.
    """
    class Logger(object):
        def __init__(self, log_file):
            self.terminal = sys.stdout
            self.log_file = open(log_file, "a", encoding="utf-8")

        def write(self, message):
            self.terminal.write(message)
            self.log_file.write(message)

        def flush(self):
            self.terminal.flush()
            self.log_file.flush()

    log_file_dir = os.path.dirname(log_file_path)
    if not os.path.exists(log_file_dir):
        os.makedirs(log_file_dir)

    sys.stdout = Logger(log_file_path)

MAX_ABSOLUTE_POSITION = 100
MAX_ORDER_SIZE = 50
MAX_OPEN_ORDER = 50
OUTSTANDING_VOLUME = 1000

class OpenOrders:
    def __init__(self, contract):
        self.contract_name = contract
        self.price_to_id = {}
        self.id_to_price = {}
        self.id_to_qty = {}
        self.id_to_timestamp = {}
        self.id_to_filled_prices = {}

    def adjust_qty(self, id, adj):
        self.id_to_qty[id] += adj
        if self.id_to_qty[id] == 0:
            if id in self.id_to_price:
                price = self.id_to_price[id]
                del self.id_to_price[id]
                if price in self.price_to_id:
                    del self.price_to_id[price]
                del self.id_to_qty[id]
                del self.id_to_timestamp[id]

    def modify_order(self, price, qty, old_id, new_id):
        if old_id == new_id:
            if old_id not in self.id_to_price:
                self.id_to_price[old_id] = price
                self.price_to_id[price] = old_id
                self.id_to_qty[old_id] = qty
            else:
                old_price = self.id_to_price[old_id]
                if old_price in self.price_to_id:
                    del self.price_to_id[old_price]
                self.price_to_id[price] = old_id
                self.id_to_price[old_id] = price
                self.id_to_qty[old_id] = qty
        else:
            if old_id in self.id_to_price:
                old_price = self.id_to_price[old_id]
                if old_price in self.price_to_id:
                    del self.price_to_id[old_price]
                del self.id_to_price[old_id]
            if old_id in self.id_to_qty:
                del self.id_to_qty[old_id]
            if old_id in self.id_to_timestamp:
                del self.id_to_timestamp[old_id]
            self.id_to_price[new_id] = price
            self.price_to_id[price] = new_id
            self.id_to_qty[new_id] = qty
        self.id_to_timestamp[new_id] = time.time()

    def get_qty(self, price):
        p_id = self.price_to_id[price]
        return self.id_to_qty[p_id]

    def get_id(self, price):
        return self.price_to_id[price]

class PIPOBot(xchange_client.XChangeClient):
    def __init__(self, host, username, password):
        super().__init__(host, username, password)
        self.contracts = ["EPT", "DLO", "MKU", "IGM", "BRV"]
        self.order_size = 5
        self.order_l0_ratio, self.order_l1_ratio, self.order_l2_ratio, self.order_l3_ratio = (1, 0.5, 0.35, 0.15)
        self.edge = 1  # cent margin
        self.l0_spread = self.edge
        self.l1_spread = 2
        self.l2_spread = self.l1_spread * 2
        self.l3_spread = self.l1_spread * 3
        self.level_positions = {}
        self.order_ids = {}
        self.open_orders = {}
        self.order_id_to_price = {}
        self.order_filled_prices = {}
        for contract in self.contracts:
            self.order_filled_prices[contract] = {'L0': [], 'L1': [], 'L2': [], 'L3': []}
            self.level_positions[contract] = {'L0': 0, 'L1': 0, 'L2': 0, 'L3': 0}
            self.order_ids[contract] = {'l0_bid': '', 'l0_ask': '', 'l1_bid': '', 'l1_ask': '', 'l2_bid': '', 'l2_ask': '', 'l3_bid': '', 'l3_ask': ''}
            self.open_orders[contract] = OpenOrders(contract)
        self.fade = 1
        self.slack = 1
        self.total_open = 0
        self.total_fills = 0
        self.total_transactions = 0
        
    async def custom_place_order(self, symbol: str, qty: int, side: Side, px: int = None) -> str:
        print([self.open_orders[contract].id_to_qty.values() for contract in self.contracts])
        if sum(sum(1 for v in self.open_orders[contract].id_to_qty.values()) for contract in self.contracts) >= MAX_OPEN_ORDER:
            await self.cancel_unfavorable_orders_from_pool()
        order_id = await super().place_order(symbol, qty, side, px)
        self.order_id_to_price[order_id] = px
        return order_id
    
    async def get_penny_prices(self, contract):
        book = self.order_books[contract]
        bot_bids = {self.open_orders[contract].id_to_price[order_id] for order_id in self.order_ids[contract].values() if order_id in self.open_orders[contract].id_to_price and self.open_orders[contract].id_to_qty[order_id] > 0}
        bot_asks = {self.open_orders[contract].id_to_price[order_id] for order_id in self.order_ids[contract].values() if order_id in self.open_orders[contract].id_to_price and self.open_orders[contract].id_to_qty[order_id] < 0}
        
        valid_asks = [px for px, qty in book.asks.items() if qty > 0 and px not in bot_asks]
        valid_bids = [px for px, qty in book.bids.items() if qty > 0 and px not in bot_bids]
        
        penny_ask_price = min(valid_asks) if valid_asks else None
        penny_bid_price = max(valid_bids) if valid_bids else None
        
        return penny_bid_price, penny_ask_price
    
    async def place_level_orders(self, contract, level, spread, penny_bid_price, penny_ask_price):
        ratio = getattr(self, f'order_l{level}_ratio') / sum(getattr(self, f'order_l{i}_ratio') for i in range(4))
        order_size = min(int(self.order_size * ratio), self.order_size)
        old_bid_id = self.order_ids[contract][f'l{level}_bid']
        old_ask_id = self.order_ids[contract][f'l{level}_ask']
        bid_qty = min(order_size, MAX_ABSOLUTE_POSITION - self.positions[contract], MAX_ORDER_SIZE)
        ask_qty = min(order_size, MAX_ABSOLUTE_POSITION + self.positions[contract], MAX_ORDER_SIZE)
        bid_response = None
        ask_response = None
        
        if self.level_positions[contract][f'L{level}'] > ratio * MAX_ABSOLUTE_POSITION:
            if penny_ask_price - self.edge > self.get_avg_filled_price(contract, level, 'ask'):
                ask_response = await self.custom_place_order(contract, ask_qty, xchange_client.Side.SELL, penny_ask_price - self.edge)
                self.order_ids[contract][f'l{level}_ask'] = ask_response
                self.open_orders[contract].modify_order(penny_ask_price - self.edge, -ask_qty, old_ask_id, ask_response)
                print(f"[AGGRESSIVE SELL] {contract} - Placed Aggressive Ask Order. ID: {ask_response}, Qty: {ask_qty}, Price: {penny_ask_price - self.edge}")
        elif self.level_positions[contract][f'L{level}'] < -ratio * MAX_ABSOLUTE_POSITION:
            if penny_bid_price + self.edge < self.get_avg_filled_price(contract, level, 'bid'):
                bid_response = await self.custom_place_order(contract, bid_qty, xchange_client.Side.BUY, penny_bid_price + self.edge)
                self.order_ids[contract][f'l{level}_bid'] = bid_response
                self.open_orders[contract].modify_order(penny_bid_price + self.edge, bid_qty, old_bid_id, bid_response)
                print(f"[AGGRESSIVE BUY] {contract} - Placed Aggressive Bid Order. ID: {bid_response}, Qty: {bid_qty}, Price: {penny_bid_price + self.edge}")
        else:
            if bid_qty > 0 and penny_bid_price - spread > 0:
                bid_response = await self.custom_place_order(contract, bid_qty, xchange_client.Side.BUY, penny_bid_price - spread - self.edge)
                self.order_ids[contract][f'l{level}_bid'] = bid_response
                self.open_orders[contract].modify_order(penny_bid_price - spread - self.edge, bid_qty, old_bid_id, bid_response)
            if ask_qty > 0:
                ask_response = await self.custom_place_order(contract, ask_qty, xchange_client.Side.SELL, penny_ask_price + spread + self.edge)
                self.order_ids[contract][f'l{level}_ask'] = ask_response
                self.open_orders[contract].modify_order(penny_ask_price + spread + self.edge, -ask_qty, old_ask_id, ask_response)
    
    def get_avg_filled_price(self, contract, level, side):
        filled_prices = self.order_filled_prices[contract][f'L{level}']
        filled_prices = [price for price in filled_prices if (side == 'bid' and price > 0) or (side == 'ask' and price < 0)]
        return sum(filled_prices) / len(filled_prices) if filled_prices else (float('inf') if side == 'bid' else float('-inf'))
    
    async def cancel_unfavorable_orders(self, contract):
        while sum(1 for v in self.open_orders[contract].id_to_qty.values()) > 0.9 * MAX_OPEN_ORDER:
            oldest_order_id = min(self.open_orders[contract].id_to_timestamp, key=self.open_orders[contract].id_to_timestamp.get)
            await self.cancel_order(oldest_order_id)
            del self.open_orders[contract].id_to_timestamp[oldest_order_id]
            print(f"[CANCELLED] Oldest Order - ID: {oldest_order_id}")
    
    async def cancel_unfavorable_orders_from_pool(self):
        all_open_orders = [(contract, order_id, timestamp) for contract in self.contracts for order_id, timestamp in self.open_orders[contract].id_to_timestamp.items()]
        all_open_orders.sort(key=lambda x: x[2])  # Sort by timestamp
        while sum(sum(1 for v in self.open_orders[contract].id_to_qty.values()) for contract in self.contracts) > 0.9 * MAX_OPEN_ORDER and len(all_open_orders) > 0:
            contract, order_id, _ = all_open_orders.pop(0)
            await self.cancel_order(order_id)
            del self.open_orders[contract].id_to_timestamp[order_id]
            print(f"[CANCELLED] Oldest Order from Pool - ID: {order_id}, Contract: {contract}")
    
    async def adjust_aggressiveness(self, contract, penny_bid_price, penny_ask_price):
        print(self.positions[contract])
        if self.positions[contract] > 0.8 * MAX_ABSOLUTE_POSITION:
            ask_qty = min(self.order_size, MAX_ABSOLUTE_POSITION + self.positions[contract], MAX_ORDER_SIZE)
            if penny_ask_price - self.edge > self.get_avg_filled_price(contract, 0, 'ask'):
                ask_response = await self.custom_place_order(contract, ask_qty, xchange_client.Side.SELL, penny_ask_price - self.edge)
                self.order_ids[contract]['l0_ask'] = ask_response
                print(f"[AGGRESSIVE ASK] {contract} - Placed Aggressive Ask Order. ID: {ask_response}, Qty: {ask_qty}, Price: {penny_ask_price - self.edge}")
        elif self.positions[contract] < -0.5 * MAX_ABSOLUTE_POSITION:  # Adjust the threshold for aggressive buying
            bid_qty = min(2 * self.order_size, MAX_ABSOLUTE_POSITION - self.positions[contract], MAX_ORDER_SIZE)  # Increase the bid quantity
            if penny_bid_price + self.edge < self.get_avg_filled_price(contract, 0, 'bid'):
                bid_response = await self.custom_place_order(contract, bid_qty, xchange_client.Side.BUY, penny_bid_price + self.edge)
                self.order_ids[contract]['l0_bid'] = bid_response
                print(f"[AGGRESSIVE BUY] {contract} - Placed Aggressive Bid Order. ID: {bid_response}, Qty: {bid_qty}, Price: {penny_bid_price + self.edge}")

    async def trade(self):
        print("[INFO] Starting Quote Update Loop")
        while True:
            iteration_orders = 0
            iteration_fills = 0
            for contract in self.contracts:
                penny_bid_price, penny_ask_price = await self.get_penny_prices(contract)
                if penny_bid_price is not None and penny_ask_price is not None:
                    print(f"[INFO] {contract} - Penny Bid: {penny_bid_price}, Penny Ask: {penny_ask_price}")
                    if penny_ask_price - penny_bid_price > self.slack:
                        for level in range(0, 4):
                            spread = getattr(self, f'l{level}_spread')
                            await self.place_level_orders(contract, level, spread, penny_bid_price, penny_ask_price)
                    print(f"[LOG] {contract} - {self.level_positions[contract]}")
                    print(f"[SUMMARY] Iteration Orders: {iteration_orders}, Iteration Fills: {iteration_fills}, Total Orders: {self.total_open}, Total Fills: {self.total_fills}, Total Transactions: {self.total_transactions}")
                    await self.adjust_aggressiveness(contract, penny_bid_price, penny_ask_price)
                    await self.cancel_unfavorable_orders(contract)
                else:
                    print(f"[WARNING] {contract} - Invalid Penny Prices. Skipping order placement.")
            await asyncio.sleep(1)
    
    async def bot_handle_book_update(self, symbol):
        pass
    
    async def bot_handle_trade_msg(self, symbol, price, qty):
        pass
    
    async def bot_handle_order_fill(self, order_id, qty, price):
        for contract in self.contracts:
            open_orders = self.open_orders[contract]
            for order_key, order_value in self.order_ids[contract].items():
                if order_id == order_value:
                    if order_id in open_orders.id_to_qty:
                        filled_qty = -qty if qty > 0 else qty
                        open_orders.adjust_qty(order_id, filled_qty)
                        self.positions[contract] += qty
                        if 'l0' in order_key:
                            self.level_positions[contract]['L0'] += qty
                            self.order_filled_prices[contract]['L0'].append(price if qty > 0 else -price)
                        if 'l1' in order_key:
                            self.level_positions[contract]['L1'] += qty
                            self.order_filled_prices[contract]['L1'].append(price if qty > 0 else -price)
                        elif 'l2' in order_key:
                            self.level_positions[contract]['L2'] += qty
                            self.order_filled_prices[contract]['L2'].append(price if qty > 0 else -price)
                        elif 'l3' in order_key:
                            self.level_positions[contract]['L3'] += qty
                            self.order_filled_prices[contract]['L3'].append(price if qty > 0 else -price)
                        print(f"[FILL] Order Fill - {contract}: {'+'if qty > 0 else ''}{qty} @ {price}")
                    break
    
    async def bot_handle_order_rejected(self, order_id, reason):
        print(f"[REJECTED] Order Rejected - Order ID: {order_id}, Reason: {reason}")
        await self.cancel_unfavorable_orders_from_pool()
    
    async def bot_handle_cancel_response(self, order_id, success, error):
        if success:
            print(f"[CANCELED] Order Cancelled - Order ID: {order_id}")
        else:
            print(f"[ERROR] Order Cancellation Failed - Order ID: {order_id}, Error: {error}")
    
    async def start(self):
        asyncio.create_task(self.trade())
        await self.connect()

async def main():
    log_file_path = "/Users/divy/Desktop/Chicago-Trading-Competition-2024/case_1/log/file.txt"
    log_to_file(log_file_path)
    bot = PIPOBot("staging.uchicagotradingcompetition.com:3333", "university_of_chicago_umassamherst", "ekans-mew-8133")
    try:
        await bot.start()
        await asyncio.Event().wait()
    except Exception as e:
        print(f"Exception occurred: {e}")
        print("Restarting the bot...")
        await asyncio.sleep(1)  # Wait for a short duration before restarting
    except KeyboardInterrupt:
        print("KeyboardInterrupt: Closing the event loop...")

if __name__ == "__main__":
    asyncio.run(main())