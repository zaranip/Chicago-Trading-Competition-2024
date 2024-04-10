from xchangelib import xchange_client
from xchangelib.xchange_client import Side
import asyncio
import time
import sys
import os
import random

random.seed(1)

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
MAX_ORDER_SIZE = 40
MAX_OPEN_ORDER = 10
MAX_OUTSTANDING_VOLUME = 120

class OpenOrders:
    def __init__(self, contract):
        self.contract_name = contract
        self.level_positions = {level: 0 for level in ['L0', 'L1', 'L2', 'L3']}
        self.order_ids = {f'{level}_{side}': '' for level in ['L0', 'L1', 'L2', 'L3'] for side in ['bid', 'ask']}
        self.order_filled_prices = {level: {'bid': [], 'ask': []} for level in ['L0', 'L1', 'L2', 'L3']}
        self.id_to_price = {}
        self.id_to_qty = {}
        self.id_to_timestamp = {}

    def adjust_qty(self, id, adj):
        self.id_to_qty[id] += adj
        if self.id_to_qty[id] == 0:
            self.remove_order(id)

    def remove_order(self, id):
        if id in self.id_to_price:
            del self.id_to_price[id]
        if id in self.id_to_qty:
            del self.id_to_qty[id]
        if id in self.id_to_timestamp:
            del self.id_to_timestamp[id]

    def modify_order(self, price, qty, old_id, new_id):
        self.remove_order(old_id)
        self.id_to_price[new_id] = price
        self.id_to_qty[new_id] = qty
        self.id_to_timestamp[new_id] = time.time()

class PIPOBot(xchange_client.XChangeClient):
    def __init__(self, host, username, password):
        super().__init__(host, username, password)
        self.contracts = ["EPT", "DLO", "MKU", "IGM", "BRV"]
        self.order_size = 10
        self.order_ratios = {'L0': 1, 'L1': 0.5, 'L2': 0.35, 'L3': 0.15}
        self.margin = 1  # cent margin
        self.spreads = {'L0': self.margin, 'L1': 2, 'L2': 4, 'L3': 6}
        self.open_orders = {contract: OpenOrders(contract) for contract in self.contracts}
        self.slack = 1
        
    async def custom_place_order(self, symbol: str, qty: int, side: Side, px: int = None) -> str:
        # Place an order and handle max open orders limit per contract
        if len(self.open_orders[symbol].id_to_qty) >= MAX_OPEN_ORDER:
            await self.cancel_unfavorable_orders_from_contract(symbol)
        order_id = await super().place_order(symbol, qty, side, px)
        return order_id
    
    async def get_penny_prices(self, contract):
        # Get the best bid and ask prices for penny jumping
        book = self.order_books[contract]
        bot_bids = {self.open_orders[contract].id_to_price[order_id] for order_id in self.open_orders[contract].order_ids.values() if order_id in self.open_orders[contract].id_to_price and self.open_orders[contract].id_to_qty[order_id] > 0}
        bot_asks = {self.open_orders[contract].id_to_price[order_id] for order_id in self.open_orders[contract].order_ids.values() if order_id in self.open_orders[contract].id_to_price and self.open_orders[contract].id_to_qty[order_id] < 0}
        valid_asks = [px for px, qty in book.asks.items() if qty > 0 and px not in bot_asks]
        valid_bids = [px for px, qty in book.bids.items() if qty > 0 and px not in bot_bids]
        penny_ask_price = min(valid_asks) if valid_asks else None
        penny_bid_price = max(valid_bids) if valid_bids else None
        return penny_bid_price, penny_ask_price
    
    def get_avg_filled_price(self, contract, level, side, qty):
        # Calculate the average filled price for a given level and side
        filled_prices = self.open_orders[contract].order_filled_prices[level][side]
        if len(filled_prices) == 0:
            return None
        if qty <= 0:
            return None
        sampled_prices = random.sample(filled_prices, min(qty, len(filled_prices)))
        return sum(sampled_prices) / len(sampled_prices)

    def remove_filled_prices(self, contract, level, side, qty):
        # Remove the filled prices used for average calculation
        filled_prices = self.open_orders[contract].order_filled_prices[level][side]
        if len(filled_prices) > qty:
            sampled_prices = random.sample(filled_prices, min(qty, len(filled_prices)))
            for price in sampled_prices:
                filled_prices.remove(price)
    
    async def place_level_orders(self, contract, level, spread, penny_bid_price, penny_ask_price):
        # Place orders at each level based on ratios and manage aggressive orders
        ratio = self.order_ratios[level] / sum(self.order_ratios.values())
        order_size = min(int(self.order_size * ratio), self.order_size)
        old_bid_id = self.open_orders[contract].order_ids[f'{level}_bid']
        old_ask_id = self.open_orders[contract].order_ids[f'{level}_ask']
        
        # Calculate outstanding volume for bids and asks for the contract
        outstanding_bid_volume = sum(qty for qty in self.open_orders[contract].id_to_qty.values() if qty > 0)
        outstanding_ask_volume = sum(abs(qty) for qty in self.open_orders[contract].id_to_qty.values() if qty < 0)
        
        # Adjust bid and ask quantities based on outstanding volume and position limits
        bid_qty = min(order_size, MAX_ABSOLUTE_POSITION - self.positions[contract], MAX_ORDER_SIZE, MAX_OUTSTANDING_VOLUME - outstanding_bid_volume)
        ask_qty = min(order_size, MAX_ABSOLUTE_POSITION + self.positions[contract], MAX_ORDER_SIZE, MAX_OUTSTANDING_VOLUME - outstanding_ask_volume)
        
        # Aggressive buying and selling
        await self.adjust_aggressiveness_ratio(contract, level, 'bid', penny_bid_price)
        await self.adjust_aggressiveness_ratio(contract, level, 'ask', penny_ask_price)
        
        # Normal penny in penny out strategy
        if bid_qty > 0 and penny_bid_price - spread > 0:
            bid_response = await self.custom_place_order(contract, bid_qty, xchange_client.Side.BUY, penny_bid_price + spread)
            self.open_orders[contract].order_ids[f'{level}_bid'] = bid_response
            self.open_orders[contract].modify_order(penny_bid_price + spread, bid_qty, old_bid_id, bid_response)
        if ask_qty > 0:
            ask_response = await self.custom_place_order(contract, ask_qty, xchange_client.Side.SELL, penny_ask_price - spread)
            self.open_orders[contract].order_ids[f'{level}_ask'] = ask_response
            self.open_orders[contract].modify_order(penny_ask_price - spread, -ask_qty, old_ask_id, ask_response)

    async def cancel_unfavorable_orders_from_contract(self, contract):
        # Cancel unfavorable orders for a specific contract based on max open orders limit
        while len(self.open_orders[contract].id_to_qty) >= MAX_OPEN_ORDER or sum(self.open_orders[contract].id_to_qty.values()) > MAX_OUTSTANDING_VOLUME:
            oldest_order_id = min(self.open_orders[contract].id_to_timestamp, key=self.open_orders[contract].id_to_timestamp.get)
            await self.cancel_order(oldest_order_id)
            self.open_orders[contract].remove_order(oldest_order_id)
    
    async def adjust_aggressiveness_ratio(self, contract, level, side, penny_price):
        # Adjust aggressiveness based on level positions and overall position
        ratio = self.order_ratios[level] / sum(self.order_ratios.values())
        maintain_ratio_qty = abs(abs(self.open_orders[contract].level_positions[level]) - int(self.order_size * ratio))
        avg_filled_price = self.get_avg_filled_price(contract, level, "bid" if side=="ask" else "ask", maintain_ratio_qty)
        #print(f"[INFO] {contract} - Level: {level}, Side: {side}, Qty: {qty}, Penny Price: {penny_price}, Avg Filled Price: {avg_filled_price}")
        
        if avg_filled_price is None:
            return
        
        if self.open_orders[contract].level_positions[level] < -ratio * MAX_ABSOLUTE_POSITION:
            print(f"[TRY BUY] {contract} - buy: {maintain_ratio_qty} - penny_bid_price: {penny_price+self.margin}, avg_filled_price: {avg_filled_price}")
            if penny_price < avg_filled_price:
                diff = avg_filled_price - penny_price
                bid_response = await self.custom_place_order(contract, maintain_ratio_qty, xchange_client.Side.BUY, penny_price + int(diff/2))
                self.open_orders[contract].order_ids[f'{level}_bid'] = bid_response
                self.open_orders[contract].modify_order(penny_price + self.margin, maintain_ratio_qty, self.open_orders[contract].order_ids[f'{level}_bid'], bid_response)
                self.remove_filled_prices(contract, level, 'bid', maintain_ratio_qty)
                print(f"[AGGRESSIVE BUY] {contract} - Placed Aggressive Bid Order. ID: {bid_response}, Qty: {maintain_ratio_qty}, Price: {penny_price + self.margin}")
        
        elif self.open_orders[contract].level_positions[level] > ratio * MAX_ABSOLUTE_POSITION:
            print(f"[TRY ASK] {contract} - ask: {maintain_ratio_qty} - penny_ask_price: {penny_price - self.margin}, avg_filled_price: {avg_filled_price}")
            if penny_price - self.margin > avg_filled_price:
                diff = penny_price - avg_filled_price
                ask_response = await self.custom_place_order(contract, maintain_ratio_qty, xchange_client.Side.SELL, penny_price - int(diff/2))
                self.open_orders[contract].order_ids[f'{level}_ask'] = ask_response
                self.open_orders[contract].modify_order(penny_price - self.margin, -maintain_ratio_qty, self.open_orders[contract].order_ids[f'{level}_ask'], ask_response)
                self.remove_filled_prices(contract, level, 'ask', maintain_ratio_qty)
                print(f"[AGGRESSIVE SELL] {contract} - Placed Aggressive Ask Order. ID: {ask_response}, Qty: {maintain_ratio_qty}, Price: {penny_price - self.margin}")
        
    
    async def adjust_aggressiveness_position(self, contract, level, side, penny_price):
        # Adjust aggressiveness based on level positions and overall position
        qty = abs(abs(self.positions[contract]) - 0.5 * MAX_ABSOLUTE_POSITION)
        avg_filled_price = self.get_avg_filled_price(contract, level, "bid" if side=="ask" else "ask", qty)
        #print(f"[INFO] {contract} - Level: {level}, Side: {side}, Qty: {qty}, Penny Price: {penny_price}, Avg Filled Price: {avg_filled_price}")
        
        if avg_filled_price is None:
            return
        
        if self.positions[contract] > 0.5 * MAX_ABSOLUTE_POSITION:
            print(f"[TRY ASK] {contract} - ask: {qty} - penny_ask_price: {penny_price - self.margin}, avg_filled_price: {avg_filled_price}")
            if penny_price > avg_filled_price:
                diff = penny_price - avg_filled_price
                ask_response = await self.custom_place_order(contract, qty, xchange_client.Side.SELL, penny_price - int(diff/2))
                self.open_orders[contract].order_ids[f'{level}_ask'] = ask_response
                self.remove_filled_prices(contract, level, 'ask', qty)
                print(f"[AGGRESSIVE ASK] {contract} - Placed Aggressive Ask Order. ID: {ask_response}, Qty: {qty}, Price: {penny_price - int(diff/2)}")
        
        elif self.positions[contract] < -0.5 * MAX_ABSOLUTE_POSITION:
            print(f"[TRY BUY] {contract} - buy: {qty} - penny_bid_price: {penny_price + self.margin}, avg_filled_price: {avg_filled_price}")
            if penny_price < avg_filled_price:
                diff = avg_filled_price - penny_price
                bid_response = await self.custom_place_order(contract, qty, xchange_client.Side.BUY, penny_price + int(diff/2))
                self.open_orders[contract].order_ids[f'{level}_bid'] = bid_response
                self.remove_filled_prices(contract, level, 'bid', qty)
                print(f"[AGGRESSIVE BUY] {contract} - Placed Aggressive Bid Order. ID: {bid_response}, Qty: {qty}, Price: {penny_price + int(diff/2)}")



    async def trade(self):
        # Main trading loop
        print("[INFO] Starting Quote Update Loop")
        while True:
            pos = {}
            for contract in self.contracts:
                penny_bid_price, penny_ask_price = await self.get_penny_prices(contract)
                if penny_bid_price is not None and penny_ask_price is not None:
                    if penny_ask_price - penny_bid_price > self.slack:
                        for level in self.order_ratios:
                            spread = self.spreads[level]
                            await self.place_level_orders(contract, level, spread, penny_bid_price, penny_ask_price)
                await self.adjust_aggressiveness_position(contract, level, 'bid', penny_bid_price)
                await self.adjust_aggressiveness_position(contract, level, 'ask', penny_ask_price)
                pos[contract] = self.positions[contract]
            print(f"[LOG] Total Positions: {pos}")
            await asyncio.sleep(1)
        
    async def bot_handle_order_fill(self, order_id, qty, price):
        # Handle order fills
        for contract in self.contracts:
            open_orders = self.open_orders[contract]
            for order_key, order_value in open_orders.order_ids.items():
                if order_id == order_value:
                    if order_id in open_orders.id_to_qty:
                        filled_qty = qty
                        open_orders.adjust_qty(order_id, -filled_qty)
                        level = order_key.split('_')[0]
                        side = 'bid' if 'bid' in order_key else 'ask'
                        self.open_orders[contract].order_filled_prices[level][side].append(price)
                        print(f"[FILL] Order Fill - {contract}: {filled_qty} @ {price}")
                        if order_id in open_orders.id_to_qty and open_orders.id_to_qty[order_id] == 0:
                            open_orders.remove_order(order_id)
                    break

    async def bot_handle_order_rejected(self, order_id, reason):
        # Handle order rejections and cancel unfavorable orders from the contract
        for contract in self.contracts:
            if order_id in self.open_orders[contract].id_to_qty:
                await self.cancel_unfavorable_orders_from_contract(contract)
                break

    async def bot_handle_trade_msg(self, symbol: str, price: int, qty: int):
        # Handle trade messages
        pass

    async def bot_handle_cancel_response(self, order_id, success, error):
        # Handle cancel responses
        if success:
            print(f"[CANCELED] Order Cancelled - Order ID: {order_id}")
        else:
            print(f"[ERROR] Order Cancellation Failed - Order ID: {order_id}, Error: {error}")

    async def start(self):
        # Start the bot
        asyncio.create_task(self.trade())
        await self.connect()

async def main():
    log_file_path = "/Users/divy/Desktop/Chicago-Trading-Competition-2024/case_1/log/file.txt"
    log_to_file(log_file_path)
    bot = PIPOBot("staging.uchicagotradingcompetition.com:3333", "university_of_chicago_umassamherst", "ekans-mew-8133")
    await bot.start()
    await asyncio.Event().wait()

if __name__ == "__main__":
    asyncio.run(main())