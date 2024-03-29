Usage
=====

.. _installation:
Installation
------------

To use xchangelib, first install it using pip:

.. code-block:: console

   $ pip install xchangelib


.. _setup:
Setup
------------
To use the client, create a subclass of the **XChangeClient** object.
 .. code-block:: python

    class MyXchangeClient(xchange_client.XChangeClient):
        '''A shell client with the methods that can be implemented to interact with the xchange.'''

         def __init__(self, host: str, username: str, password: str):
            super().__init__(host, username, password)

         async def start(self):
             """
             Creates tasks that can be run in the background. Then connects to the exchange
             and listens for messages.
             """
             await self.connect()

You should also implement the bot handler methods that are defined. Below we implemented the methods
so that it prints the type of update and some information about the update when the bot receives an update from the
exchange. In your bots, you can choose to trade or cancel orders based on the messages received.

.. code-block:: python

   class MyXchangeClient(xchange_client.XChangeClient):
        ...
        async def bot_handle_cancel_response(self, order_id: str, success: bool, error: Optional[str]) -> None:
            order = self.open_orders[order_id]
            print(f"{'Market' if order[2] else 'Limit'} Order ID {order_id} cancelled, {order[1]} unfilled")

        async def bot_handle_order_fill(self, order_id: str, qty: int, price: int):
            print("order fill", self.positions)

        async def bot_handle_order_rejected(self, order_id: str, reason: str) -> None:
            print("order rejected because of ", reason)

        async def bot_handle_trade_msg(self, symbol: str, price: int, qty: int):
            print("something was traded")

        async def bot_handle_book_update(self, symbol: str) -> None:
            print("book update")

        async def bot_handle_swap_response(self, swap: str, qty: int, success: bool):
            print("Swap response")

The next step is to connect your bot to the exchange. To do this, instantiate your bot and
call the start function. Now, the bot will connect to the xchange and your bot handlers will
be run whenever a message is received from the xchange.

.. code-block:: python

    async def main():
        SERVER = 'SERVER URL'
        my_client = MyXchangeClient(SERVER,"USERNAME","PASSWORD")
        await my_client.start()
        return

    if __name__ == "__main__":
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(main())

Your bot can also choose to operate independent of function callbacks from the xchange. To
do this, you can use **asyncio.create_task** to create a task before you start your bot. For example,
we created a function below that prints the order books every 3 seconds.

.. code-block:: python

    async def view_books(self):
        """Prints the books every 3 seconds."""
        while True:
            await asyncio.sleep(3)
            for security, book in self.order_books.items():
                sorted_bids = sorted((k,v) for k,v in book.bids.items() if v != 0)
                sorted_asks = sorted((k,v) for k,v in book.asks.items() if v != 0)
                print(f"Bids for {security}:\n{sorted_bids}")
                print(f"Asks for {security}:\n{sorted_asks}")

We also need to modify our **start(self)** function to create the tasks.

.. code-block:: python

     async def start(self):
         """
         Creates tasks that can be run in the background. Then connects to the exchange
         and listens for messages.
         """
         asyncio.create_task(self.view_books())
         await self.connect()


The **XChangeClient** that we subclass also has a number of helpful methods
implemented to interact with the xchange. You can place and cancel orders, view
your positions, view the order books, and place swaps. Check out the :doc:`api` for more
information about the helper functions.

.. code-block:: python

    async def trade(self):
        """This is a task that is started right before the bot connects and runs in the background."""
        await asyncio.sleep(5)
        print("attempting to trade")
        await self.place_order("BRV",3, xchange_client.Side.SELL, 7)

        # Cancelling an order
        order_to_cancel = await self.place_order("BRV",3, xchange_client.Side.BUY, 5)
        await asyncio.sleep(5)
        await self.cancel_order(order_to_cancel)

        # Placing Swap requests
        await self.place_swap_order('toJAK', 1)
        await asyncio.sleep(5)
        await self.place_swap_order('fromSCP', 1)
        await asyncio.sleep(5)

        # Placing an order that gets rejected for exceeding qty limits
        await self.place_order("BRV",1000, xchange_client.Side.SELL, 7)
        await asyncio.sleep(5)

        # Placing a market order
        market_order_id = await self.place_order("BRV",10, xchange_client.Side.SELL)
        print("Market Order ID:", market_order_id)
        await asyncio.sleep(5)

        # Viewing Positions
        print("My positions:", self.positions)


You can look at the entire example bot in  **example.py**