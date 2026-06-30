import pandas as pd

# ---------------------------------------------------
# 1. Load price data from Nat_Gas.csv
# ---------------------------------------------------

price_df = pd.read_csv("data/Nat_Gas.csv")
price_df["Dates"] = pd.to_datetime(price_df["Dates"])


# ---------------------------------------------------
# 2. Pricing function
# ---------------------------------------------------

def price_storage_contract(
    price_df,
    injection_schedule,
    withdrawal_schedule,
    inj_rate,
    wdr_rate,
    max_volume,
    monthly_storage_fee,
    inj_fee_per_mmbtu=0.0,
    wdr_fee_per_mmbtu=0.0,
    transport_fee_inj=0.0,
    transport_fee_wdr=0.0,
):
    """
    Prototype pricing model for a gas storage contract.

    Parameters
    ----------
    price_df : pd.DataFrame
        Must contain columns:
            - 'Dates'  (datetime-like)
            - 'Prices' (float, $/MMBtu)

    injection_schedule : dict
        Keys: dates (str or datetime) on which injection occurs.
        Values: volumes to inject on that date (in MMBtu).

    withdrawal_schedule : dict
        Keys: dates (str or datetime) on which withdrawal occurs.
        Values: volumes to withdraw on that date (in MMBtu).

    inj_rate : float
        Maximum volume that can be injected in a single period (MMBtu).

    wdr_rate : float
        Maximum volume that can be withdrawn in a single period (MMBtu).

    max_volume : float
        Maximum storage capacity (MMBtu).

    monthly_storage_fee : float
        Fixed storage fee per time period (per date in price_df), in $.

    inj_fee_per_mmbtu : float, optional
        Variable fee per MMBtu injected, in $/MMBtu.

    wdr_fee_per_mmbtu : float, optional
        Variable fee per MMBtu withdrawn, in $/MMBtu.

    transport_fee_inj : float, optional
        Fixed transport fee whenever an injection happens on a date, in $.

    transport_fee_wdr : float, optional
        Fixed transport fee whenever a withdrawal happens on a date, in $.

    Returns
    -------
    total_value : float
        Present value of all cash flows (since interest rates are 0).

    cash_flows_df : pd.DataFrame
        Table with columns:
            - 'date'
            - 'price'
            - 'cash_flow'
            - 'inventory'
    """

    # Normalize keys to Timestamps for easy comparison
    inj_sched = {pd.to_datetime(k): float(v) for k, v in injection_schedule.items()}
    wdr_sched = {pd.to_datetime(k): float(v) for k, v in withdrawal_schedule.items()}

    inventory = 0.0
    total_value = 0.0
    records = []

    # Sort price data by date
    df_sorted = price_df.sort_values("Dates")

    for _, row in df_sorted.iterrows():
        date = row["Dates"]
        price = float(row["Prices"])
        cf = 0.0  # cash flow on this date

        # 1) Pay storage fee every period (fixed fee)
        if monthly_storage_fee:
            cf -= monthly_storage_fee

        # 2) Injection
        if date in inj_sched:
            vol_req = inj_sched[date]

            # Enforce injection rate
            if vol_req > inj_rate + 1e-8:
                raise ValueError(
                    f"Injection rate exceeded on {date.date()}: "
                    f"requested {vol_req}, max {inj_rate}"
                )

            # Enforce storage capacity
            if inventory + vol_req > max_volume + 1e-8:
                raise ValueError(
                    f"Storage capacity exceeded on {date.date()}: "
                    f"inventory {inventory}, inject {vol_req}, max {max_volume}"
                )

            volume = vol_req

            # Cash flows: buy gas + injection fee + transport fee
            cf -= volume * price                         # purchase cost
            cf -= inj_fee_per_mmbtu * volume             # variable injection fee
            cf -= transport_fee_inj                      # fixed transport per injection date

            # Update inventory
            inventory += volume

        # 3) Withdrawal
        if date in wdr_sched:
            vol_req = wdr_sched[date]

            # Enforce withdrawal rate
            if vol_req > wdr_rate + 1e-8:
                raise ValueError(
                    f"Withdrawal rate exceeded on {date.date()}: "
                    f"requested {vol_req}, max {wdr_rate}"
                )

            # Check that we have enough gas to withdraw
            if vol_req > inventory + 1e-8:
                raise ValueError(
                    f"Not enough inventory to withdraw on {date.date()}: "
                    f"inventory {inventory}, withdraw {vol_req}"
                )

            volume = vol_req

            # Cash flows: sell gas - withdrawal fee - transport fee
            cf += volume * price                         # sale revenue
            cf -= wdr_fee_per_mmbtu * volume             # variable withdrawal fee
            cf -= transport_fee_wdr                      # fixed transport per withdrawal date

            # Update inventory
            inventory -= volume

        total_value += cf

        records.append(
            {
                "date": date,
                "price": price,
                "cash_flow": cf,
                "inventory": inventory,
            }
        )

    cash_flows_df = pd.DataFrame(records)
    return total_value, cash_flows_df


# ---------------------------------------------------
# 3. TESTS: example scenarios
# ---------------------------------------------------

if __name__ == "__main__":

    # ========= Test 1: Very simple example (single injection & withdrawal) =========
    # Let's say:
    # - Inject 1,000,000 MMBtu on 2021-05-31
    # - Withdraw 1,000,000 MMBtu on 2023-12-31
    # - Max volume = 1,000,000
    # - Rates are enough to handle that volume in one go.
    # - Monthly storage fee = 100,000
    # - No injection/withdrawal/transport fees in this test.

    injection_schedule_1 = {"2021-05-31": 1_000_000}
    withdrawal_schedule_1 = {"2023-12-31": 1_000_000}

    total_value_1, cf_table_1 = price_storage_contract(
        price_df=price_df,
        injection_schedule=injection_schedule_1,
        withdrawal_schedule=withdrawal_schedule_1,
        inj_rate=1_000_000,
        wdr_rate=1_000_000,
        max_volume=1_000_000,
        monthly_storage_fee=100_000,
        inj_fee_per_mmbtu=0.0,
        wdr_fee_per_mmbtu=0.0,
        transport_fee_inj=0.0,
        transport_fee_wdr=0.0,
    )

    print("=== Test 1: Simple single injection & withdrawal ===")
    print("Contract value:", total_value_1)
    print(cf_table_1.head(), "...\n")

    # ========= Test 2: Include injection/withdrawal and transport fees =========
    # Suppose:
    # - Same dates & volumes as Test 1.
    # - Injection/withdrawal fee: $10,000 per 1 million MMBtu -> 0.01 $/MMBtu
    # - Transport fee: $50,000 for each injection & each withdrawal

    inj_fee_per_mmbtu = 10_000 / 1_000_000  # = 0.01 $/MMBtu
    wdr_fee_per_mmbtu = 10_000 / 1_000_000  # = 0.01 $/MMBtu

    total_value_2, cf_table_2 = price_storage_contract(
        price_df=price_df,
        injection_schedule=injection_schedule_1,
        withdrawal_schedule=withdrawal_schedule_1,
        inj_rate=1_000_000,
        wdr_rate=1_000_000,
        max_volume=1_000_000,
        monthly_storage_fee=100_000,
        inj_fee_per_mmbtu=inj_fee_per_mmbtu,
        wdr_fee_per_mmbtu=wdr_fee_per_mmbtu,
        transport_fee_inj=50_000,
        transport_fee_wdr=50_000,
    )

    print("=== Test 2: With fees and transport ===")
    print("Contract value:", total_value_2)
    print(cf_table_2.head(), "...\n")

    # ========= Test 3: Multiple injections and withdrawals =========
    # Example:
    # - Inject 0.5 million in 2021-05-31 and another 0.5 million in 2021-06-30
    # - Withdraw 0.5 million in 2022-12-31 and another 0.5 million in 2023-01-31
    # - Keep same capacity and costs.

    injection_schedule_3 = {
        "2021-05-31": 500_000,
        "2021-06-30": 500_000,
    }

    withdrawal_schedule_3 = {
        "2022-12-31": 500_000,
        "2023-01-31": 500_000,
    }

    total_value_3, cf_table_3 = price_storage_contract(
        price_df=price_df,
        injection_schedule=injection_schedule_3,
        withdrawal_schedule=withdrawal_schedule_3,
        inj_rate=600_000,
        wdr_rate=600_000,
        max_volume=1_000_000,
        monthly_storage_fee=80_000,
        inj_fee_per_mmbtu=0.005,   # 5k per million injected
        wdr_fee_per_mmbtu=0.005,   # 5k per million withdrawn
        transport_fee_inj=30_000,
        transport_fee_wdr=30_000,
    )

    print("=== Test 3: Multiple injections & withdrawals ===")
    print("Contract value:", total_value_3)
    print(cf_table_3.head(), "...")
