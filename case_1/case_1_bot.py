import collections
from datetime import datetime
from typing import Optional
from xchangelib import xchange_client
from  prediction import Prediction
import asyncio
import numpy as np
import pandas as pd


# constants
MAX_ORDER_SIZE = 100
MAX_OPEN_ORDERS = 100
OUTSTANDING_VOLUME = 100
MAX_ABSOLUTE_POSITION = 100


class OrderResponse:
    def __init__(self, order_id: str):
        self.order_id = order_id

class OpenOrders:
    def __init__(self, contract):
        self.contract_name = contract
        self.num_open_orders = 0
        self.price_to_id = {}
        self.id_to_price = {}
        self.id_to_qty = {}

    def adjust_qty(self, id, adj):
        self.id_to_qty[id] += adj
        if self.id_to_qty[id] == 0:
            self.num_open_orders -= 1
            price = self.id_to_price[id]
            del self.id_to_price[id]
            del self.price_to_id[price]
            del self.id_to_qty[id]

    def modify_order(self, price, qty, old_id, new_id):
        if old_id == new_id:
            if old_id not in self.id_to_price:
                self.id_to_price[old_id] = price
                self.price_to_id[price] = old_id
                self.id_to_qty[old_id] = qty
                self.num_open_orders += 1
            else:
                del self.price_to_id[self.id_to_price[old_id]]
                self.price_to_id[price] = old_id
                self.id_to_price[old_id] = price
                self.id_to_qty[old_id] = qty
        else:
            if old_id not in self.id_to_price:
                self.id_to_price[new_id] = price
                self.price_to_id[price] = new_id
                self.id_to_qty[new_id] = qty
                self.num_open_orders += 1
            else:
                del self.price_to_id[self.id_to_price[old_id]]
                del self.id_to_price[old_id]
                del self.id_to_qty[old_id]
                self.price_to_id[price] = new_id
                self.id_to_price[new_id] = price
                self.id_to_qty[new_id] = qty

    def get_qty(self, price):
        p_id = self.price_to_id[price]
        return self.id_to_qty[p_id]

    def get_id(self, price):
        return self.price_to_id[price]

class MainBot(xchange_client.XChangeClient):
    '''A shell client with the methods that can be implemented to interact with the xchange.'''

    def __init__(self, host: str, username: str, password: str):
        super().__init__(host, username, password)
        self.contracts = ["EPT", "DLO", "MKU", "IGM", "BRV"]
        self.order_size = 10
        self.level_orders = 10
        self.l1_spread = 20
        self.l2_spread = self.l1_spread * 2
        self.l3_spread = self.l1_spread * 3
        self.order_ids = collections.defaultdict()
        self.old_order_ids = collections.defaultdict()
        self.open_orders = collections.defaultdict(int)
        self.open_level_orders = collections.defaultdict(int)
        self.outstanding_volume = collections.defaultdict(int)
        self.ladder = collections.defaultdict(dict)
        
            # self.open_orders[contract] = OpenOrders(contract)

    async def bot_handle_cancel_response(self, order_id: str, success: bool, error: Optional[str]) -> None:
        if success:
            symbol, side, level, qty, value = self.order_ids[order_id]
            self.outstanding_volume[symbol] -= qty
            self.open_orders[symbol] -= 1
            self.open_level_orders[symbol] -= 1 if level else 0
            del self.order_ids[order_id]
            with open(f"./log/filled/round_data_{start_time}.txt", "a") as f:
                f.write(f"[CANCELLED] {order_id} {(symbol, side)}\n")
        else:
            print(f"[DEBUG] Order Cancellation Failed - Order ID: {order_id}, Error: {error}")

    async def bot_handle_order_fill(self, order_id: str, qty: int, price: int):
        global start_time
        symbol, side, level, vol, value = self.order_ids[order_id]
        self.outstanding_volume[symbol] -= qty
        self.open_orders[symbol] -= 1
        self.open_level_orders[symbol] -= 1 if level else 0
        gap = value - price
        if vol == qty:
            del self.order_ids[order_id]
            if order_id in self.old_order_ids:
                self.old_order_ids.remove(order_id)
        else:
            self.order_ids[order_id] = (symbol, side, level, vol - qty, value)

        with open(f"./log/filled/round_data_{start_time}.txt", "a") as f:
            f.write(f"{order_id} {(symbol, side)} {qty} {price} {gap}\n")

        

    async def bot_handle_order_rejected(self, order_id: str, reason: str) -> None:
        symbol, side, level, qty, price = self.order_ids[order_id]
        self.outstanding_volume[symbol] -= qty
        self.open_orders[symbol] -= 1
        self.open_level_orders[symbol] -= 1 if level else 0
        del self.order_ids[order_id]
        if order_id in self.old_order_ids:
            self.old_order_ids.remove(order_id)
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

    async def bot_place_order(self, symbol, qty, side, price, level=False, aggressive=False):
        vol = min(qty, OUTSTANDING_VOLUME - self.outstanding_volume[symbol], 
                  abs(MAX_ABSOLUTE_POSITION - self.positions[symbol]) if side == xchange_client.Side.BUY else abs(-MAX_ABSOLUTE_POSITION - self.positions[symbol]))
        
            
        if self.open_orders[symbol] < MAX_OPEN_ORDERS and vol > 0:
            order_id = await self.place_order(symbol, vol, side, price)
            
            self.order_ids[order_id] = (symbol, "BID" if side == xchange_client.Side.BUY else "ASK", level, vol, price)
            self.open_orders[symbol] += 1
            self.outstanding_volume[symbol] += vol

            if aggressive and vol < qty:
                # will cancel whatever oldest order and place this order
                pass

            with open(f"./log/placed/round_data_{start_time}.txt", "a") as f:
                f.write(f"{order_id} {symbol} {price}\n")

            return order_id

    async def load_my_positions(self):
        pass

    async def trade(self):
        """This is a task that is started right before the bot connects and runs in the background."""
        await self.load_my_positions()
        symbols = ["EPT","DLO","MKU","IGM","BRV"]
        etfs = ["SCP", "JAK"]
        df = pd.read_csv("Case1_Historical_Amended.csv")
        predictors = [Prediction(symbol, df[symbol].to_numpy()) for symbol in symbols + etfs]

        while True:
            volume = dict((symbol, [self.positions[symbol]//10 + 1, self.positions[symbol]//10 + 1]) for symbol in symbols + etfs)
            old_order_ids = list(self.old_order_ids.keys())
            for order_id in old_order_ids:
                if order_id not in self.order_ids:
                    continue
                symbol, side, level, qty, value = self.order_ids[order_id]
                volume[symbol][0] += qty if side == "BID" and symbol in symbols else 0
                volume[symbol][1] += qty if side == "ASK" and symbol in symbols else 0
                await self.cancel_order(order_id)
                

            self.old_order_ids = self.order_ids
            
            for pred in predictors:
                order_book = self.order_books[pred.name()] if pred.name() in self.order_books else xchange_client.OrderBook()
                pred.update(order_book)
            predictions = dict((pred.name(), pred.predict(2)) for pred in predictors)
            predictions["SCP"] = (3 * predictions["EPT"] + 3*predictions["IGM"] + 4*predictions["BRV"])/10
            predictions["JAK"] = (2 * predictions['EPT'] + 5*predictions['DLO'] + 3*predictions['MKU'])/10
            bids = dict((pred.name(), pred.bid(predictions[pred.name()])) for pred in predictors)
            asks = dict((pred.name(), pred.ask(predictions[pred.name()])) for pred in predictors)

            # ETF Arbitrage

            # how aggressively to arbitrage
            rate = 0.8
            for etf in etfs:
                margin = 20
                etf_bids = sorted([(k,v) for k, v in self.order_books[etf].bids.items() if k > predictions[etf] + margin and v > 0], reverse=True)
                etf_asks = sorted((k,v) for k, v in self.order_books[etf].asks.items() if k < predictions[etf] - margin and v > 0)
                for k,v in etf_bids:
                    await self.bot_place_order(etf, int(rate * v), xchange_client.Side.SELL, k)
                for k,v in etf_asks:
                    await self.bot_place_order(etf, int(rate * v), xchange_client.Side.BUY, k)
            # Other arbitrage
            for symbol in symbols:
                margin = 50
                symbol_bids = sorted([(k,v) for k, v in self.order_books[symbol].bids.items() if k > predictions[symbol] + margin and v > 0], reverse=True)
                symbol_asks = sorted((k,v) for k, v in self.order_books[symbol].asks.items() if k < predictions[symbol] - margin and v > 0)
                for k,v in symbol_bids:
                    await self.bot_place_order(etf, v, xchange_client.Side.SELL, k)
                for k,v in symbol_asks:
                    await self.bot_place_order(etf, v, xchange_client.Side.BUY, k)
            
            # Normal Trading
            for symbol, _ in predictions.items():
                if symbol in symbols:
                    if int(bids[symbol]) > 0:
                        await self.bot_place_order(symbol, volume[symbol][0], xchange_client.Side.BUY, int(bids[symbol]))
                    elif int(asks[symbol]) > 0:
                        await self.bot_place_order(symbol, volume[symbol][1], xchange_client.Side.SELL, int(asks[symbol])) 

            
                
            # Level Orders
            for symbol in symbols:
                if self.open_orders[symbol] < MAX_OPEN_ORDERS and self.outstanding_volume[symbol] < OUTSTANDING_VOLUME:
                    for level in range(1, 4):
                        if bids[symbol] < 0 or asks[symbol] < 0:
                            continue
                        spread = getattr(self, f"l{level}_spread")
                        bid = bids[symbol] - spread
                        ask = asks[symbol] + spread

                        aggressive = level < 2
                        vol = min(OUTSTANDING_VOLUME - self.outstanding_volume[symbol], self.level_orders)
                        if self.open_level_orders[symbol] < self.level_orders:
                            await self.bot_place_order(symbol, vol, xchange_client.Side.BUY, int(bid), True, aggressive)
                            self.open_level_orders[symbol] += 1
                        if self.open_level_orders[symbol] < self.level_orders:
                            await self.bot_place_order(symbol, vol, xchange_client.Side.SELL, int(ask), True, aggressive)
                            self.open_level_orders[symbol] += 1
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
            print(f"Exception occurred: {e}")
            print("Restarting the bot...")
            await asyncio.sleep(1)  # Wait for a short duration before restarting
        except KeyboardInterrupt:
            print("KeyboardInterrupt: Closing the event loop...")
            break

if __name__ == "__main__":
    start_time = datetime.now().strftime("%y-%m-%d-%H-%M-%S")
    # while True:
    asyncio.run(main())
    


