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

class OrderResponse:
    def __init__(self, order_id: str):
        self.order_id = order_id

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
        self.order_size = 10
        self.order_l1_ratio, self.order_l2_ratio, self.order_l3_ratio = (5,3,2)
        self.l1_spread = 2
        self.l2_spread = self.l1_spread * 2
        self.l3_spread = self.l1_spread * 3
        self.order_ids = {}
        self.open_orders = {}
        self.order_id_to_price = {}
        for contract in self.contracts:
            self.order_ids[contract] = {'bid': '', 'ask': '', 'l1_bid': '', 'l1_ask': '', 'l2_bid': '', 'l2_ask': '', 'l3_bid': '', 'l3_ask': ''}
            self.open_orders[contract] = OpenOrders(contract)

    async def close(self):
        await self.disconnect()
        await super().close()
        
    async def custom_place_order(self, symbol: str, qty: int, side: Side, px: int = None) -> OrderResponse:
        order_id = await super().place_order(symbol, qty, side, px)
        self.order_id_to_price[order_id] = px
        return OrderResponse(order_id)

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
            
            bid_qty = min(self.order_size, MAX_ABSOLUTE_POSITION - self.positions[contract])
            ask_qty = min(self.order_size, MAX_ABSOLUTE_POSITION + self.positions[contract])
            
            bid_response = None
            ask_response = None
            if bid_qty > 0:
                bid_response = await self.custom_place_order(contract, bid_qty, xchange_client.Side.BUY, round(penny_bid_price, 2))
                self.order_ids[contract]['bid'] = bid_response.order_id
                self.open_orders[contract].modify_order(round(penny_bid_price, 2), bid_qty, old_bid_id, bid_response.order_id)
            if ask_qty > 0:
                ask_response = await self.custom_place_order(contract, ask_qty, xchange_client.Side.SELL, round(penny_ask_price, 2))
                self.order_ids[contract]['ask'] = ask_response.order_id
                self.open_orders[contract].modify_order(round(penny_ask_price, 2), -ask_qty, old_ask_id, ask_response.order_id)
            if bid_response is not None and ask_response is not None:
                print(f"[DEBUG] {contract} - Placed Penny Orders. Bid ID: {bid_response.order_id}, Ask ID: {ask_response.order_id}")
            elif bid_response is not None:
                print(f"[DEBUG] {contract} - Placed Penny Order. Bid ID: {bid_response.order_id}")
            elif ask_response is not None:
                print(f"[DEBUG] {contract} - Placed Penny Order. Ask ID: {ask_response.order_id}")

    async def place_level_orders(self, contract, level, spread, penny_bid_price, penny_ask_price):
        if abs(self.positions[contract]) < MAX_ABSOLUTE_POSITION:
            await self.adjust_level_ratios(contract)
            
            old_bid_id = self.order_ids[contract][f'l{level}_bid']
            old_ask_id = self.order_ids[contract][f'l{level}_ask']
            
            total_ratio = sum(getattr(self, f'order_l{i}_ratio') for i in range(1, 4))
            order_size = min(int(self.order_size * getattr(self, f'order_l{level}_ratio') / total_ratio), self.order_size)
            
            bid_qty = min(order_size, MAX_ABSOLUTE_POSITION - self.positions[contract])
            ask_qty = min(order_size, MAX_ABSOLUTE_POSITION + self.positions[contract])
            
            bid_response = None
            ask_response = None
            if bid_qty > 0:
                bid_response = await self.custom_place_order(contract, bid_qty, xchange_client.Side.BUY, round(penny_bid_price - spread, 2))
                self.order_ids[contract][f'l{level}_bid'] = bid_response.order_id
                self.open_orders[contract].modify_order(round(penny_bid_price - spread, 2), bid_qty, old_bid_id, bid_response.order_id)
            if ask_qty > 0:
                ask_response = await self.custom_place_order(contract, ask_qty, xchange_client.Side.SELL, round(penny_ask_price + spread, 2))
                self.order_ids[contract][f'l{level}_ask'] = ask_response.order_id
                self.open_orders[contract].modify_order(round(penny_ask_price + spread, 2), -ask_qty, old_ask_id, ask_response.order_id)
            if bid_response is not None and ask_response is not None:
                print(f"[DEBUG] {contract} - Placed L{level} Orders. Bid ID: {bid_response.order_id}, Ask ID: {ask_response.order_id}")
            elif bid_response is not None:
                print(f"[DEBUG] {contract} - Placed L{level} Order. Bid ID: {bid_response.order_id}")
            elif ask_response is not None:
                print(f"[DEBUG] {contract} - Placed L{level} Order. Ask ID: {ask_response.order_id}")

    async def adjust_level_ratios(self, contract):
        target_ratios = {1: self.order_l1_ratio, 2: self.order_l2_ratio, 3: self.order_l3_ratio}
        current_ratios = {level: 0 for level in range(1, 4)}
        
        for level in range(1, 4):
            bid_key = f'l{level}_bid'
            ask_key = f'l{level}_ask'
            if self.order_ids[contract][bid_key] != '':
                current_ratios[level] += 1
            if self.order_ids[contract][ask_key] != '':
                current_ratios[level] += 1
        
        print(f"[DEBUG] {contract} - Current L1:L2:L3 Ratio: {current_ratios[1]}:{current_ratios[2]}:{current_ratios[3]}")
        
        for level in range(1, 4):
            while current_ratios[level] > target_ratios[level]:
                bid_key = f'l{level}_bid'
                ask_key = f'l{level}_ask'
                oldest_bid_id = self.order_ids[contract][bid_key]
                oldest_ask_id = self.order_ids[contract][ask_key]
                
                if oldest_bid_id != '' and oldest_ask_id != '':
                    if oldest_bid_id in self.open_orders[contract].id_to_timestamp and oldest_ask_id in self.open_orders[contract].id_to_timestamp:
                        if self.open_orders[contract].id_to_timestamp[oldest_bid_id] < self.open_orders[contract].id_to_timestamp[oldest_ask_id]:
                            await self.cancel_order(oldest_bid_id)
                            current_ratios[level] -= 1
                        else:
                            await self.cancel_order(oldest_ask_id)
                            current_ratios[level] -= 1
                    elif oldest_bid_id in self.open_orders[contract].id_to_timestamp:
                        await self.cancel_order(oldest_bid_id)
                        current_ratios[level] -= 1
                    elif oldest_ask_id in self.open_orders[contract].id_to_timestamp:
                        await self.cancel_order(oldest_ask_id)
                        current_ratios[level] -= 1


    async def trade(self):
        print("[DEBUG] Starting Quote Update Loop")
        while True:
            symbols = ["EPT", "DLO", "MKU", "IGM", "BRV"]
    etfs = ["SCP", "JAK"]
    df = pd.read_csv("Case1_Historical.csv")
    predictors = [Prediction(symbol, df[symbol].to_numpy()) for symbol in symbols]

    while True:
        k = 2
        for pred in predictors:
            order_book = self.order_books[pred.name()] if pred.name() in self.order_books else xchange_client.OrderBook()
            pred.update(order_book)
        predictions = dict((pred.name(), pred.predict(k)) for pred in predictors)
        bids = dict((pred.name(), pred.bid(predictions[pred.name()])) for pred in predictors)
        asks = dict((pred.name(), pred.ask(predictions[pred.name()])) for pred in predictors)
        for symbol, _ in predictions.items():
            await self.bot_place_order(symbol, 5, xchange_client.Side.BUY, int(bids[symbol]))
            await self.bot_place_order(symbol, 5, xchange_client.Side.SELL, int(asks[symbol]))

        # ETF Arbitrage
        for etf in etfs:
            margin = 20
            if etf == "SCP":
                price = (3 * predictions["EPT"] + 3 * predictions["IGM"] + 4 * predictions["BRV"]) / 10
            elif etf == "JAK":
                price = (2 * predictions['EPT'] + 5 * predictions['DLO'] + 3 * predictions['MKU']) / 10
            etf_bids = sorted((k, v) for k, v in self.order_books[etf].bids.items() if k > price + margin and v > 0)
            etf_asks = sorted((k, v) for k, v in self.order_books[etf].asks.items() if k < price - margin and v > 0)
            for k, v in etf_bids:
                await self.bot_place_order(etf, v, xchange_client.Side.SELL, k)
            for k, v in etf_asks:
                await self.bot_place_order(etf, v, xchange_client.Side.BUY, k)
                
            for contract in self.contracts:
                print(f"[DEBUG] {contract} - Updating Quotes")
                penny_bid_price, penny_ask_price = await self.get_penny_prices(contract)

                if penny_bid_price is not None and penny_ask_price is not None:
                    print(f"[DEBUG] {contract} - Penny Bid: {penny_bid_price}, Penny Ask: {penny_ask_price}")

                    if penny_ask_price - penny_bid_price > 0:
                        await self.place_penny_orders(contract, penny_bid_price, penny_ask_price)

                        for level in range(1, 4):
                            spread = getattr(self, f'l{level}_spread')
                            if penny_bid_price is not None and penny_bid_price - spread > 0:
                                await self.place_level_orders(contract, level, spread, penny_bid_price, penny_ask_price)
                    
                    # Check if the stock has reached the maximum or minimum threshold
                    if self.positions[contract] >= MAX_ABSOLUTE_POSITION:
                        # Sell off the most unfavorable bid at a one-cent profit
                        unfavorable_bid_id = min(self.order_ids[contract]['bid'], self.order_ids[contract]['l1_bid'], self.order_ids[contract]['l2_bid'], self.order_ids[contract]['l3_bid'], key=lambda x: self.order_id_to_price.get(x, float('inf')))
                        if unfavorable_bid_id in self.order_id_to_price:
                            unfavorable_bid_price = self.order_id_to_price[unfavorable_bid_id]
                            await self.cancel_order(unfavorable_bid_id)
                            await self.custom_place_order(contract, 1, xchange_client.Side.SELL, round(unfavorable_bid_price + 1, 2))
                    elif self.positions[contract] <= -MAX_ABSOLUTE_POSITION:
                        # Buy the most unfavorable ask at a one-cent profit
                        unfavorable_ask_id = max(self.order_ids[contract]['ask'], self.order_ids[contract]['l1_ask'], self.order_ids[contract]['l2_ask'], self.order_ids[contract]['l3_ask'], key=lambda x: self.order_id_to_price.get(x, float('-inf')))
                        if unfavorable_ask_id in self.order_id_to_price:
                            unfavorable_ask_price = self.order_id_to_price[unfavorable_ask_id]
                            await self.cancel_order(unfavorable_ask_id)
                            await self.custom_place_order(contract, 1, xchange_client.Side.BUY, round(unfavorable_ask_price - 1, 2))

                    print(self.order_ids[contract])
                else:
                    print(f"[DEBUG] {contract} - Invalid Penny Prices. Skipping order placement.")

                # Print the number of l1, l2, l3 bids and asks
                l1_bids = sum(1 for order_id in self.order_ids[contract].values() if 'l1_bid' in order_id)
                l1_asks = sum(1 for order_id in self.order_ids[contract].values() if 'l1_ask' in order_id)
                l2_bids = sum(1 for order_id in self.order_ids[contract].values() if 'l2_bid' in order_id)
                l2_asks = sum(1 for order_id in self.order_ids[contract].values() if 'l2_ask' in order_id)
                l3_bids = sum(1 for order_id in self.order_ids[contract].values() if 'l3_bid' in order_id)
                l3_asks = sum(1 for order_id in self.order_ids[contract].values() if 'l3_ask' in order_id)

                print(f"[DEBUG] {contract} - L1 Bids: {l1_bids}, L1 Asks: {l1_asks}, L2 Bids: {l2_bids}, L2 Asks: {l2_asks}, L3 Bids: {l3_bids}, L3 Asks: {l3_asks}")

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
        #print(f"[DEBUG] Trade Message - Symbol: {symbol}, Price: {price}, Quantity: {qty}")
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
                            print(f"[DEBUG] Order Fill - {contract}: +{qty} @ {price}")
                        else:
                            open_orders.adjust_qty(order_id, qty)
                            self.positions[contract] -= qty
                            print(f"[DEBUG] Order Fill - {contract}: {qty} @ {price}")
                        
                        # Update the id_to_filled_prices dictionary
                        if order_id not in open_orders.id_to_filled_prices:
                            open_orders.id_to_filled_prices[order_id] = []
                        open_orders.id_to_filled_prices[order_id].extend([price] * abs(qty))
                    break

    async def bot_handle_order_rejected(self, order_id, reason):
        print(f"[DEBUG] Order Rejected - Order ID: {order_id}, Reason: {reason}")

    async def bot_handle_cancel_response(self, order_id, success, error):
        if success:
            print(f"[DEBUG] Order Cancelled - Order ID: {order_id}")
        else:
            print(f"[DEBUG] Order Cancellation Failed - Order ID: {order_id}, Error: {error}")

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
    except KeyboardInterrupt:
        print("KeyboardInterrupt: Closing the event loop...")
    finally:
        await bot.close()
    
if __name__ == "__main__":
    asyncio.run(main())





