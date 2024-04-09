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
MAX_ORDER_SIZE = 30
MAX_OPEN_ORDER = 100
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

        self.order_size = 30

        self.order_l0_ratio, self.order_l1_ratio, self.order_l2_ratio, self.order_l3_ratio = (1,0.5,0.35,0.15)
        self.l1_spread = 2
        self.l2_spread = self.l1_spread * 2
        self.l3_spread = self.l1_spread * 3

        self.order_ids = {}
        self.open_orders = {}
        self.order_id_to_price = {}
        for contract in self.contracts:
            self.order_ids[contract] = {'bid': '', 'ask': '', 'l1_bid': '', 'l1_ask': '', 'l2_bid': '', 'l2_ask': '', 'l3_bid': '', 'l3_ask': ''}
            self.open_orders[contract] = OpenOrders(contract)

        self.edge = 0
        self.fade = 0
        self.slack = 0
        
    async def custom_place_order(self, symbol: str, qty: int, side: Side, px: int = None) -> str:
        order_id = await super().place_order(symbol, qty, side, px)
        self.order_id_to_price[order_id] = px
        return order_id

    async def get_penny_prices(self, contract):
        book = self.order_books[contract]
        bot_bids = {self.open_orders[contract].id_to_price[order_id] for order_id in self.order_ids[contract].values() if order_id in self.open_orders[contract].id_to_price and self.open_orders[contract].id_to_qty[order_id] > 0}
        bot_asks = {self.open_orders[contract].id_to_price[order_id] for order_id in self.order_ids[contract].values() if order_id in self.open_orders[contract].id_to_price and self.open_orders[contract].id_to_qty[order_id] < 0}
        
        penny_ask_price = None
        penny_bid_price = None
        
        if book.asks:
            valid_asks = [px for px, qty in book.asks.items() if qty > 0 and px not in bot_asks]
            if valid_asks:
                penny_ask_price = min(valid_asks) - 1
        
        if book.bids:
            valid_bids = [px for px, qty in book.bids.items() if qty > 0 and px not in bot_bids]
            if valid_bids:
                penny_bid_price = max(valid_bids) + 1
        
        return penny_bid_price, penny_ask_price

    async def place_penny_orders(self, contract, penny_bid_price, penny_ask_price):
        if abs(self.positions[contract]) < MAX_ABSOLUTE_POSITION:
            old_bid_id = self.order_ids[contract]['bid']
            old_ask_id = self.order_ids[contract]['ask']
            
            ratio = self.order_l0_ratio / sum(getattr(self, f'order_l{i}_ratio') for i in range(4))
            order_size = min(int(self.order_size * ratio), self.order_size)
            
            bid_qty = min(order_size, MAX_ABSOLUTE_POSITION - self.positions[contract])
            ask_qty = min(order_size, MAX_ABSOLUTE_POSITION + self.positions[contract])
            
            bid_response = None
            ask_response = None
            if bid_qty > 0 and bid_qty <= MAX_OPEN_ORDER and sum(abs(v) for v in self.open_orders[contract].id_to_qty.values()) + bid_qty <= OUTSTANDING_VOLUME:
                bid_response = await self.custom_place_order(contract, bid_qty, xchange_client.Side.BUY, penny_bid_price - self.edge - int(self.fade * self.positions[contract] / 100))
                self.order_ids[contract]['bid'] = bid_response
                self.open_orders[contract].modify_order(penny_bid_price - self.edge - int(self.fade * self.positions[contract] / 100), bid_qty, old_bid_id, bid_response)
            if ask_qty > 0 and ask_qty <= MAX_OPEN_ORDER and sum(abs(v) for v in self.open_orders[contract].id_to_qty.values()) + ask_qty <= OUTSTANDING_VOLUME:
                ask_response = await self.custom_place_order(contract, ask_qty, xchange_client.Side.SELL, penny_ask_price + self.edge + int(self.fade * self.positions[contract] / 100))
                self.order_ids[contract]['ask'] = ask_response
                self.open_orders[contract].modify_order(penny_ask_price + self.edge + int(self.fade * self.positions[contract] / 100), -ask_qty, old_ask_id, ask_response)
            if bid_response is not None and ask_response is not None:
                print(f"{bid_response} ('{contract}', 'BID') {bid_qty} {penny_bid_price - self.edge - int(self.fade * self.positions[contract] / 100)} 0 {ask_response} ('{contract}', 'ASK') {ask_qty} {penny_ask_price + self.edge + int(self.fade * self.positions[contract] / 100)} 0")
            elif bid_response is not None:
                print(f"{bid_response} ('{contract}', 'BID') {bid_qty} {penny_bid_price - self.edge - int(self.fade * self.positions[contract] / 100)} 0")
            elif ask_response is not None:
                print(f"{ask_response} ('{contract}', 'ASK') {ask_qty} {penny_ask_price + self.edge + int(self.fade * self.positions[contract] / 100)} 0")

    async def place_level_orders(self, contract, level, spread, penny_bid_price, penny_ask_price):
        ratio = getattr(self, f'order_l{level}_ratio') / sum(getattr(self, f'order_l{i}_ratio') for i in range(4))
        order_size = min(int(self.order_size * ratio), self.order_size)
        
        if abs(self.positions[contract]) >= ratio * MAX_ABSOLUTE_POSITION:
            await self.adjust_level_ratios(contract, level)
        else:
            old_bid_id = self.order_ids[contract][f'l{level}_bid']
            old_ask_id = self.order_ids[contract][f'l{level}_ask']
            
            bid_qty = min(order_size, MAX_ABSOLUTE_POSITION - self.positions[contract])
            ask_qty = min(order_size, MAX_ABSOLUTE_POSITION + self.positions[contract])
            
            bid_response = None
            ask_response = None
            if bid_qty > 0 and penny_bid_price - spread > 0 and bid_qty <= MAX_OPEN_ORDER and sum(abs(v) for v in self.open_orders[contract].id_to_qty.values()) + bid_qty <= OUTSTANDING_VOLUME:
                bid_response = await self.custom_place_order(contract, bid_qty, xchange_client.Side.BUY, penny_bid_price - spread - self.edge - int(self.fade * self.positions[contract] / 100))
                self.order_ids[contract][f'l{level}_bid'] = bid_response
                self.open_orders[contract].modify_order(penny_bid_price - spread - self.edge - int(self.fade * self.positions[contract] / 100), bid_qty, old_bid_id, bid_response)
            if ask_qty > 0 and ask_qty <= MAX_OPEN_ORDER and sum(abs(v) for v in self.open_orders[contract].id_to_qty.values()) + ask_qty <= OUTSTANDING_VOLUME:
                ask_response = await self.custom_place_order(contract, ask_qty, xchange_client.Side.SELL, penny_ask_price + spread + self.edge + int(self.fade * self.positions[contract] / 100))
                self.order_ids[contract][f'l{level}_ask'] = ask_response
                self.open_orders[contract].modify_order(penny_ask_price + spread + self.edge + int(self.fade * self.positions[contract] / 100), -ask_qty, old_ask_id, ask_response)
            if bid_response is not None and ask_response is not None:
                print(f"{bid_response} ('{contract}', 'L{level}_BID') {bid_qty} {penny_bid_price - spread - self.edge - int(self.fade * self.positions[contract] / 100)} 0 {ask_response} ('{contract}', 'L{level}_ASK') {ask_qty} {penny_ask_price + spread + self.edge + int(self.fade * self.positions[contract] / 100)} 0")
            elif bid_response is not None:
                print(f"{bid_response} ('{contract}', 'L{level}_BID') {bid_qty} {penny_bid_price - spread - self.edge - int(self.fade * self.positions[contract] / 100)} 0")
            elif ask_response is not None:
                print(f"{ask_response} ('{contract}', 'L{level}_ASK') {ask_qty} {penny_ask_price + spread + self.edge + int(self.fade * self.positions[contract] / 100)} 0")

    async def adjust_level_ratios(self, contract, level):
        bid_key = f'l{level}_bid'
        ask_key = f'l{level}_ask'
        
        if self.positions[contract] >= 0:
            # Remove the most unfavorable ask
            if self.order_ids[contract][ask_key]:
                unfavorable_ask_id = min(self.order_ids[contract][ask_key], key=lambda x: self.order_id_to_price.get(x, float('-inf')))
                if unfavorable_ask_id in self.order_id_to_price:
                    await self.cancel_order(unfavorable_ask_id)
                    print(f"[CANCELLED] {unfavorable_ask_id} ('{contract}', '{ask_key.upper()}')")
        else:
            # Remove the most unfavorable bid
            if self.order_ids[contract][bid_key]:
                unfavorable_bid_id = max(self.order_ids[contract][bid_key], key=lambda x: self.order_id_to_price.get(x, float('inf')))
                if unfavorable_bid_id in self.order_id_to_price:
                    await self.cancel_order(unfavorable_bid_id)
                    print(f"[CANCELLED] {unfavorable_bid_id} ('{contract}', '{bid_key.upper()}')")

    async def trade(self):
        print("[INFO] Starting Quote Update Loop")
        while True:
            
            for contract in self.contracts:
                penny_bid_price, penny_ask_price = await self.get_penny_prices(contract)

                if penny_bid_price is not None and penny_ask_price is not None:
                    print(f"[INFO] {contract} - Penny Bid: {penny_bid_price}, Penny Ask: {penny_ask_price}")

                    if penny_ask_price - penny_bid_price > self.slack:
                        await self.place_penny_orders(contract, penny_bid_price, penny_ask_price)

                        for level in range(1, 4):
                            spread = getattr(self, f'l{level}_spread')
                            await self.place_level_orders(contract, level, spread, penny_bid_price, penny_ask_price)
                    
                    #print(f"[INFO] {contract} - Current Order IDs: {self.order_ids[contract]}")
                else:
                    print(f"[WARNING] {contract} - Invalid Penny Prices. Skipping order placement.")

            await asyncio.sleep(1)

    def calculate_pnl_score(self):
        pnl = 0
        for contract in self.contracts:
            if self.order_books[contract].bids and self.order_books[contract].asks:
                mid_price = (list(self.order_books[contract].bids.keys())[-1] + list(self.order_books[contract].asks.keys())[0]) / 2
                pnl += self.positions[contract] * mid_price
        pnl += self.positions['cash']
        return pnl

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
                        if qty > 0:
                            open_orders.adjust_qty(order_id, -qty)
                            self.positions[contract] += qty
                            print(f"[FILL] Order Fill - {contract}: +{qty} @ {price}")
                        else:
                            open_orders.adjust_qty(order_id, qty)
                            self.positions[contract] -= qty
                            print(f"[FILL] Order Fill - {contract}: {qty} @ {price}")
                        
                        # Update the id_to_filled_prices dictionary
                        if order_id not in open_orders.id_to_filled_prices:
                            open_orders.id_to_filled_prices[order_id] = []
                        open_orders.id_to_filled_prices[order_id].extend([price] * abs(qty))
                    break

    async def bot_handle_order_rejected(self, order_id, reason):
        print(f"[WARNING] Order Rejected - Order ID: {order_id}, Reason: {reason}")

    async def bot_handle_cancel_response(self, order_id, success, error):
        if success:
            print(f"[INFO] Order Cancelled - Order ID: {order_id}")
        else:
            print(f"[ERROR] Order Cancellation Failed - Order ID: {order_id}, Error: {error}")

    async def start(self):
        asyncio.create_task(self.trade())
        await self.connect()


async def main():
    log_file_path = "/Users/divy/Desktop/Chicago-Trading-Competition-2024/case_1/log/file.txt"
    log_to_file(log_file_path)

    # while True:
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





