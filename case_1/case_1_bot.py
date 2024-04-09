import asyncio
import traceback
import numpy as np
import pandas as pd
import collections

from datetime import datetime
from typing import Optional
from xchangelib import xchange_client
from  prediction import Prediction


# constants
MAX_ORDER_SIZE = 100
MAX_OPEN_ORDERS = 100
OUTSTANDING_VOLUME = 100
MAX_ABSOLUTE_POSITION = 100
SYMBOLS = ["EPT", "DLO", "MKU", "IGM", "BRV"]
ETFS = ["SCP", "JAK"]
df = pd.read_csv("Case1_Historical_Amended.csv")

class OrderResponse:
    def __init__(self, order_id: str):
        self.order_id = order_id

class OpenOrders:
    def __init__(self):
        self.num_open_orders = 0
        self.outstanding_volume = 0
        self.id_to_price = {}
        self.id_to_qty = {}
        self.id_to_side = {}
        self.id_to_level = {}
        self.id_to_symbol = {}
        self.level_orders = dict((symbol, {0: 0, 1: 0, 2: 0, 3: 0}) for symbol in SYMBOLS + ETFS)
        self.queue = dict((symbol, collections.deque()) for symbol in SYMBOLS + ETFS)
    def adjust_qty(self, id, adj):
        self.id_to_qty[id] += adj
        self.outstanding_volume += adj
        if self.id_to_qty[id] == 0:
            self.remove_order(id)
    def get_price(self, id):
        return self.id_to_price[id]
    def get_num_open_orders(self):
        return self.num_open_orders
    def get_qty(self, id):
        return self.id_to_qty[id]
    def get_symbol(self, id):
        return self.id_to_symbol[id]
    def get_side(self, id):
        return self.id_to_side[id]
    def get_level(self, id):
        return self.id_to_level[id]
    def get_symbol_levels(self, symbol):
        return self.level_orders[symbol]
    def get_outstanding_volume(self):
        return self.outstanding_volume
    def get_k_oldest_order(self, symbol, k):
        if len(self.queue) >= k:
            return [self.queue[symbol][i] for i in range(k)]
        return self.queue
    def add_order(self, symbol, price, qty, id, side, level):
        self.id_to_price[id] = price
        self.id_to_qty[id] = qty
        self.id_to_side[id] = side
        self.id_to_level[id] = level
        self.id_to_symbol[id] = symbol
        self.num_open_orders += 1
        self.outstanding_volume += qty
        self.level_orders[symbol][level] += 1
        self.queue[symbol].append(id)

    def remove_order(self, id):
        level = self.id_to_level[id]
        symbol = self.id_to_symbol[id]
        self.outstanding_volume -= self.id_to_qty[id]
        self.num_open_orders -= 1
        self.level_orders[symbol][level] -= 1
        self.queue[symbol].remove(id)
        del self.id_to_price[id]
        del self.id_to_qty[id]
        del self.id_to_side[id]
        del self.id_to_level[id]
        

class MainBot(xchange_client.XChangeClient):
    '''A shell client with the methods that can be implemented to interact with the xchange.'''

    def __init__(self, host: str, username: str, password: str):
        super().__init__(host, username, password)
        self.order_size = 10
        self.level_orders = 10
        self.spreads = [20, 40, 60]
        self.open_orders_object = OpenOrders()
        self.predictors = [Prediction(symbol, df[symbol].to_numpy()) for symbol in SYMBOLS + ETFS]
        self.predictions = dict((pred.name(), 0) for pred in self.predictors)


    async def bot_handle_cancel_response(self, order_id: str, success: bool, error: Optional[str]) -> None:
        if success:
            symbol = self.open_orders_object.id_to_symbol[order_id]
            side = self.open_orders_object.get_side(order_id)
            self.open_orders_object.remove_order(order_id)
            with open(f"./log/filled/round_data_{start_time}.txt", "a") as f:
                f.write(f"[CANCELLED] {order_id} {(symbol, side)}\n")
        else:
            print(f"[DEBUG] Order Cancellation Failed - Order ID: {order_id}, Error: {error}")

    async def bot_handle_order_fill(self, order_id: str, qty: int, price: int):
        symbol = self.open_orders_object.get_symbol(order_id)
        side = self.open_orders_object.get_side(order_id)
        qty = self.open_orders_object.get_qty(order_id)
        gap = self.open_orders_object.get_price(order_id) - price
        self.open_orders_object.adjust_qty(order_id, -qty)
        with open(f"./log/filled/round_data_{start_time}.txt", "a") as f:
            f.write(f"{order_id} {(symbol, side)} {qty} {price} {gap}\n")

    async def bot_handle_order_rejected(self, order_id: str, reason: str) -> None:
        symbol = self.open_orders_object.get_symbol(order_id)
        side = self.open_orders_object.get_side(order_id)
        self.open_orders_object.remove_order(order_id)
        with open(f"./log/filled/round_data_{start_time}.txt", "a") as f:
            f.write(f"[REJECTED] {order_id} {(symbol, side)}\n")
        print(f"[DEBUG] Order Rejected - Order ID: {order_id}, Reason: {reason}")


    async def bot_handle_trade_msg(self, symbol: str, price: int, qty: int):
        # print("something was traded")
        pass

    async def bot_handle_book_update(self, symbol: str) -> None:
        # print("book update")
        pass

    async def bot_handle_swap_response(self, swap: str, qty: int, success: bool):
        # print("Swap response")
        pass

    async def bot_place_order(self, symbol, qty, side, price, level=0):      
        if not level:
            diff = self.open_orders_object.get_num_open_orders() - MAX_OPEN_ORDERS
            oldest_orders = self.open_orders_object.get_k_oldest_order(symbol, diff)
            for order_id in oldest_orders:
                await self.cancel_order(order_id)
                
        vol = min(qty,
                  MAX_ABSOLUTE_POSITION - self.positions[symbol] if side == xchange_client.Side.BUY else self.positions[symbol] + MAX_ABSOLUTE_POSITION) 
        if vol > 0:
            order_id = await self.place_order(symbol, vol, side, price)
            self.open_orders_object.add_order(symbol, price, vol, order_id, side, level)
            with open(f"./log/placed/round_data_{start_time}.txt", "a") as f:
                f.write(f"{order_id} {symbol} {price}\n")
            return order_id
        
    async def bot_place_swap_order(self):
        pass

    async def bot_place_arbitrage_order(self, symbol):
        pass

    async def bot_handle_balancing_order(self, symbol):
        pass

    async def load_my_positions(self):
        pass

    async def bot_update_predictions(self):
        self.predictions = dict((pred.name(), pred.predict(2)) for pred in self.predictors)


    async def trade(self):
        """This is a task that is started right before the bot connects and runs in the background."""
        # intended to load the position if we are disconnected somehow
        await self.load_my_positions()

        while True:
            for pred in self.predictors:
                order_book = self.order_books[pred.name()] if pred.name() in self.order_books else xchange_client.OrderBook()
                pred.update(order_book)
            bids = dict((pred.name(), pred.bid(self.predictions[pred.name()])) for pred in self.predictors)
            asks = dict((pred.name(), pred.ask(self.predictions[pred.name()])) for pred in self.predictors)

            # ETF Arbitrage
            # how aggressively to arbitrage
            rate = 0.8            
            for etf in ETFS:
                if etf == "SCP":
                    price = (3 * self.predictions["EPT"] + 3*self.predictions["IGM"] + 4*self.predictions["BRV"])/10
                elif etf == "JAK":
                    price = (2 * self.predictions['EPT'] + 5*self.predictions['DLO'] + 3*self.predictions['MKU'])/10
                margin = 20
                etf_bids = sorted([(k,v) for k, v in self.order_books[etf].bids.items() if k > price + margin and v > 0])
                etf_asks = sorted([(k,v) for k, v in self.order_books[etf].asks.items() if k < price - margin and v > 0], reverse=True)
                for k,v in etf_bids:
                    await self.bot_place_order(etf, int(rate * v), xchange_client.Side.SELL, k)
                for k,v in etf_asks:
                    await self.bot_place_order(etf, int(rate * v), xchange_client.Side.BUY, k)

            # Take advantage of the spread
            # TODO: review
            for symbol in SYMBOLS:
                margin = 50
                symbol_bids = sorted([(k,v) for k, v in self.order_books[symbol].bids.items() if k > self.predictions[symbol] + margin and v > 0])
                symbol_asks = sorted([(k,v) for k, v in self.order_books[symbol].asks.items() if k < self.predictions[symbol] - margin and v > 0], reverse=True)
                for k,v in symbol_bids:
                    await self.bot_place_order(etf, v//2+1, xchange_client.Side.SELL, k)
                for k,v in symbol_asks:
                    await self.bot_place_order(etf, v//2+1, xchange_client.Side.BUY, k)
            
            # Normal Trading
            for symbol, _ in self.predictions.items():
                if symbol in SYMBOLS:
                    if int(bids[symbol]) > 0:
                        await self.bot_place_order(symbol, abs(self.positions[symbol]//10) + 1, xchange_client.Side.BUY, int(bids[symbol]))
                    elif int(asks[symbol]) > 0:
                        await self.bot_place_order(symbol, abs(self.positions[symbol]//10) + 1, xchange_client.Side.SELL, int(asks[symbol])) 
  
            # Level Orders
            for symbol in SYMBOLS:
                for level in range(1, 4):
                    if bids[symbol] < 0 or asks[symbol] < 0:
                        continue
                    spread = self.spread[level - 1]
                    bid = bids[symbol] - spread
                    ask = asks[symbol] + spread

                    if self.open_orders_object.get_symbol_levels(symbol) < self.level_orders:
                        await self.bot_place_order(symbol, self.level_orders, xchange_client.Side.BUY, int(bid), level)
                    if self.open_orders_object.get_symbol_levels(symbol) < self.level_orders:
                        await self.bot_place_order(symbol, self.level_orders, xchange_client.Side.SELL, int(ask), level)
            # Viewing Positions
            print("My positions:", self.positions)
            await asyncio.sleep(1)

    async def view_books(self):
        """Prints the books every 3 seconds."""
        while True:
            await asyncio.sleep(3)
            for security, book in self.order_books.items():
                sorted_bids = sorted((k,v) for k,v in book.bids.items() if v != 0)
                sorted_asks = sorted((k,v) for k,v in book.asks.items() if v != 0)
                print(f"Bids for {security}:\n{sorted_bids}")
                print(f"Asks for {security}:\n{sorted_asks}")

    async def start(self):
        """
        Creates tasks that can be run in the background. Then connects to the exchange
        and listens for messages.
        """
        asyncio.create_task(self.trade())
        # asyncio.create_task(self.view_books())
        await self.connect()


async def main():
    while True:
        bot = MainBot("staging.uchicagotradingcompetition.com:3333", "university_of_chicago_umassamherst", "ekans-mew-8133")
        try:
            await bot.start()
            await asyncio.Event().wait()
        except Exception as e:
            traceback.print_exc()
            print(f"Exception occurred: {e.with_traceback(None)}")  # Print the traceback
            print("Restarting the bot...")
            # break
            await asyncio.sleep(1)  # Wait for a short duration before restarting
        except KeyboardInterrupt:
            print("KeyboardInterrupt: Closing the event loop...")
            break

if __name__ == "__main__":
    start_time = datetime.now().strftime("%y-%m-%d-%H-%M-%S")
    # while True:
    asyncio.run(main())
    


