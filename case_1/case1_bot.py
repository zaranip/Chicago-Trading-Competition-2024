from datetime import datetime
from typing import Optional
from xchangelib import xchange_client
from  prediction import Prediction
import asyncio
import numpy as np
import pandas as pd


class MyXchangeClient(xchange_client.XChangeClient):
    '''A shell client with the methods that can be implemented to interact with the xchange.'''

    def __init__(self, host: str, username: str, password: str):
        super().__init__(host, username, password)

    async def bot_handle_cancel_response(self, order_id: str, success: bool, error: Optional[str]) -> None:
        order = self.open_orders[order_id]
        print(f"{'Market' if order[2] else 'Limit'} Order ID {order_id} cancelled, {order[1]} unfilled")

    async def bot_handle_order_fill(self, order_id: str, qty: int, price: int):
        print("order fill", self.positions)

    async def bot_handle_order_rejected(self, order_id: str, reason: str) -> None:
        print("order rejected because of ", reason)


    async def bot_handle_trade_msg(self, symbol: str, price: int, qty: int):
        # print("something was traded")
        pass

    async def bot_handle_book_update(self, symbol: str) -> None:
        # print("book update")
        pass

    async def bot_handle_swap_response(self, swap: str, qty: int, success: bool):
        # print("Swap response")
        pass


    async def trade(self):
        """This is a task that is started right before the bot connects and runs in the background."""
        # await self.view_books()
        start_time = datetime.now()
        symbols = ["EPT","DLO","MKU","IGM","BRV"]
        etfs = ["JCR", "JAK"]
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
                buy_order_id = await self.place_order(symbol, 1, xchange_client.Side.BUY, int(bids[symbol]))
                sell_order_id = await self.place_order(symbol, 1, xchange_client.Side.SELL, int(asks[symbol])) 
                with open(f"./log/round_data{start_time.date()}-{str(start_time.time())[-6]}.txt", "a") as f:
                    f.write(f"{symbol}: {int(bids[symbol])}, {int(asks[symbol])}\n")
                print(symbol, int(bids[symbol]), int(asks[symbol]))
            
            # ETF Arbitrage
            for etf in etfs:
                if etf == "SCP":
                    price = (3 * predictions["EPT"] + 3*predictions["IGM"] + 4*predictions["BRV"])/10
                    scp_bids = sorted((k,v) for k, v in self.order_books["SCP"].bids.items())
                    scp_asks = sorted((k,v) for k, v in self.order_books["SCP"].asks.items())
                    if self.order_books["SCP"].bids[price] > 0:
                        buy_order_id = await self.place_order(etf, 1, xchange_client.Side.BUY, price)
                    if self.order_books["SCP"].asks[price] > 0:
                        sell_order_id = await self.place_order(etf, 1, xchange_client.Side.SELL, price)
                elif etf == "JAK":
                    predictions["JAK"] = (2 * predictions['EPT'] + 5*predictions['DLO'] + 3*predictions['MKU'])/10
                    buy_order_id = await self.place_order(etf, 1, xchange_client.Side.BUY, 1000)
                    sell_order_id = await self.place_order(etf, 1, xchange_client.Side.SELL, 1000)
                
                
            # TODO: implement the fade parameter


            # TODO: implement the selling ladder


            # Viewing Positions
            print("My positions:", self.positions)
            # await asyncio.sleep(1)

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
    SERVER = 'staging.uchicagotradingcompetition.com:3333' # run on sandbox
    my_client = MyXchangeClient(SERVER,"university_of_chicago_umassamherst","ekans-mew-8133")
    await my_client.start()
    return

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    result = loop.run_until_complete(main())

    


