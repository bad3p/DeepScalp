"""Example - How to get list of orders for 1 last hour (maximum requesting period)."""
import datetime
import logging
import os

from tinkoff.invest import Client
from tinkoff.invest.constants import INVEST_GRPC_API
from tinkoff.invest.schemas import OrderExecutionReportStatus

TOKEN = os.environ["TK_TOKEN"]

def main():
    with Client(TOKEN, target=INVEST_GRPC_API) as client:
        response = client.users.get_accounts()
        account, *_ = response.accounts
        account_id = account.id

        now = datetime.datetime.now()
        orders = client.orders.get_orders(
            account_id=account_id
        )
        print("Orders list:")
        for order in orders.orders:
            print(order)


if __name__ == "__main__":
    main()