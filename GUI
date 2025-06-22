import flet as ft
import yfinance as yf
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, AutoDateLocator
import matplotlib.ticker as mticker
from io import BytesIO
import base64
from datetime import date, timedelta, datetime

# 引入模拟日志主程序（已返回 logs, final_caps, roi）
from main import main as cli_main

def df_to_base64_plot(df: pd.DataFrame, pair: str) -> str:
    # 只绘制收盘价格曲线
    if 'Close' in df.columns:
        series = df['Close']
    else:
        series = df.iloc[:, 3]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(series.index, series.values, marker='o', markersize=4, linewidth=1)
    ax.set_title(f"{pair} (Closing Exchange Rate)", pad=10)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.xaxis.set_major_locator(AutoDateLocator(minticks=6, maxticks=10))
    ax.xaxis.set_major_formatter(DateFormatter("%m-%d"))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=6))
    fig.autofmt_xdate(rotation=0, ha="center")
    fig.tight_layout()
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

def landing_page(page: ft.Page) -> ft.Stack:
    page.bgcolor = ft.Colors.BLACK
    return ft.Stack(
        expand=True, 
        controls=[
            ft.Image(
                src="image_1.png",
                expand=True, 
                fit=ft.ImageFit.COVER,
            ),

            # 🟢 Logo in top-left corner
            
            ft.Container(
                left=20,
                top=20,
                content=ft.GestureDetector(
                on_tap=lambda e: page.go("/"),
                mouse_cursor=ft.MouseCursor.CLICK,
                content=ft.Image(
                src="Logo.png",  # your logo file in assets
                width=70,
                height=70,
                fit=ft.ImageFit.CONTAIN,
                ),
                ),
            ),

            # Foreground content in the center
            ft.Container(
                expand=True,
                alignment=ft.alignment.center,
                left=page.width * 0.18,
                top=page.height * 0.4,
                content=ft.Column(
                    [
                        ft.Text(
                            "Take off with us toward financial freedom",
                            size=54,
                            weight=ft.FontWeight.BOLD,
                            color=ft.Colors.WHITE,
                            text_align=ft.TextAlign.CENTER,
                        ),
                        ft.Text(
                            "Great results follow thoughtful decisions.",
                            size=20,
                            weight=ft.FontWeight.NORMAL,
                            color=ft.Colors.with_opacity(0.8, ft.Colors.WHITE),
                            text_align=ft.TextAlign.CENTER,
                        ),
                        ft.Container(
                            margin=ft.margin.only(top=30),
                            content=ft.ElevatedButton(
                                text="Try Simulation",
                                style=ft.ButtonStyle(
                                    bgcolor=ft.Colors.WHITE,
                                    color=ft.Colors.BLACK,
                                    shape=ft.RoundedRectangleBorder(radius=20),
                                    padding=20,
                                ),
                                on_click=lambda e: page.go("/simulator"),
                            ),
                        ),
                        ft.Text(
                            "Zero cost. Unlimited opportunity.",
                            size=14,
                            color=ft.Colors.with_opacity(0.6, ft.Colors.WHITE),
                            text_align=ft.TextAlign.CENTER,
                        )
                    ],
                    spacing=20,
                    alignment=ft.MainAxisAlignment.CENTER,
                    horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                ),
            )
        ]
    )


def fetch_fx(pair: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(pair, start=start, end=end, progress=False, auto_adjust=False)
    if df.empty:
        raise ValueError(f"No data found for {pair} in {start}–{end}")
    if isinstance(df.columns, pd.MultiIndex):
        if pair in df.columns.levels[0]:
            df = df[pair]
        else:
            df.columns = df.columns.droplevel(0)
    return df

def help_page(page: ft.Page):
    return ft.Container(
        padding=20,
        content=ft.Column(
            controls=[
                ft.Text("Simulation Help", size=30, weight=ft.FontWeight.BOLD),
                ft.Text("Learn how to use the FX Simulator and understand each metric.", size=16),
                ft.Divider(),

                ft.Text("• Currency Pair:", weight="bold"),
                ft.Text("  Dropdown to select which FX pair to view or simulate (e.g. EURUSD=X, USDJPY=X)"),

                ft.Text("• Start (YYYY-MM-DD):", weight="bold"),
                ft.Text("  Start date for fetching historical OHLC data (year–month–day)"),

                ft.Text("• End (YYYY-MM-DD):", weight="bold"),
                ft.Text("  End date for fetching historical data"),

                ft.Text("• Fetch:", weight="bold"),
                ft.Text("  Button to download the selected pair’s historical data and update the chart/table"),

                ft.Text("• Prediction Days:", weight="bold"),
                ft.Text("  Number of days to simulate forward (positive integer, max 90)"),

                ft.Text("• Predict:", weight="bold"),
                ft.Text("  Button to run the LSTM forecast and trading simulation for the specified horizon"),

                ft.Text("• <Pair> (Closing Exchange Rate):", weight="bold"),
                ft.Text("  Chart title showing the selected currency pair’s closing price series"),

                ft.Text("• Historical FX Data:", weight="bold"),
                ft.Text("  Section header indicating the table below shows the fetched historical prices"),

                ft.Text("• Date / Open / High / Low / Close:", weight="bold"),
                ft.Text("  OHLC data for each historical date"),

                ft.Text("• Predict FX Data:", weight="bold"),
                ft.Text("  Section showing the simulation’s predicted results per day"),

                ft.Text("• Day:", weight="bold"),
                ft.Text("  Simulation day number from 1 to horizon"),

                ft.Text("• Pair:", weight="bold"),
                ft.Text("  Currency pair for each row (USD/JPY, USD/EUR, USD/GBP)"),

                ft.Text("• Predicted / Actual:", weight="bold"),
                ft.Text("  Model’s forecast vs. real historical exchange rate"),

                ft.Text("• PnL (Profit & Loss):", weight="bold"),
                ft.Text("  Intraday gain/loss based on open position"),

                ft.Text("• Pos (Position):", weight="bold"),
                ft.Text("  Held position: 1 = long, -1 = short, 0 = no position"),

                ft.Text("• Final Results:", weight="bold"),
                ft.Text("  Shows account balances for each pair and total ROI after simulation"),

                ft.Divider(),
                ft.ElevatedButton(
                    text="Back to Simulator",
                    on_click=lambda e: page.go("/simulator"),
                    style=ft.ButtonStyle(padding=ft.padding.all(10), shape=ft.RoundedRectangleBorder(radius=12))
                )
            ],
            scroll=ft.ScrollMode.AUTO,
            spacing=10
        )
    )


def fx_simulator_page(page: ft.Page):
    # 界面初始化
    page.title = "FX GUI"
    page.window_width, page.window_height = 1000, 800

    # 控件：货币对、日期
    page.pair_dd = ft.Dropdown(
        label="Currency Pair", width=200,
        value="EURUSD=X",
        options=[ft.dropdown.Option(p) for p in ["EURUSD=X","USDJPY=X","GBPUSD=X","AUDUSD=X"]],
    )
    today = date.today()
    one_year_ago = today - timedelta(days=365)
    page.start_tf = ft.TextField(label="Start (YYYY-MM-DD)", width=150, value=str(one_year_ago))
    page.end_tf = ft.TextField(label="End   (YYYY-MM-DD)", width=150, value=str(today))
    fetch_btn = ft.ElevatedButton("Fetch", on_click=lambda e: on_fetch(page))

    # 控件：预测天数与按钮
    page.horizon_tf = ft.TextField(label="Prediction Days", width=100, value="50")
    predict_btn = ft.ElevatedButton("Predict", on_click=lambda e: on_predict(page))

    # 图表占位
    placeholder = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNg"
        "YAAAAAMAASsJTYQAAAAASUVORK5CYII="
    )
    page.img_chart = ft.Image(src_base64=placeholder, fit=ft.ImageFit.CONTAIN, height=400)

    # 历史数据表头
    page.header = ft.Row([
        ft.Text("Date", width=120, weight="bold"),
        ft.Text("Open", width=80, weight="bold", text_align="right"),
        ft.Text("High", width=80, weight="bold", text_align="right"),
        ft.Text("Low", width=80, weight="bold", text_align="right"),
        ft.Text("Close", width=80, weight="bold", text_align="right"),
    ], spacing=5)
    page.list_view = ft.ListView(expand=True, spacing=2, height=200)

    # 模拟结果表头
    page.sim_header = ft.Row([
        ft.Text("Day", width=50, weight="bold"),
        ft.Text("Pair", width=80, weight="bold"),
        ft.Text("Predicted", width=80, weight="bold", text_align="right"),
        ft.Text("Actual", width=80, weight="bold", text_align="right"),
        ft.Text("PnL", width=80, weight="bold", text_align="right"),
        ft.Text("Pos", width=80, weight="bold", text_align="right"),
    ], spacing=5)
    page.sim_list_view = ft.ListView(expand=True, spacing=2, height=200)

    # 最终结果显示区
    page.final_results = ft.Column(spacing=5)

    # 主内容布局（左右分栏）
    main_layout = ft.Row([
        ft.Column([
            ft.Row([page.pair_dd, page.start_tf, page.end_tf, fetch_btn], spacing=15),
            ft.Row([page.img_chart], alignment=ft.MainAxisAlignment.START),
            ft.Text("Historical FX Data", weight="bold", size=16),
            page.header,
            page.list_view,
        ], expand=3, spacing=10),

        ft.Column([
            ft.Row([page.horizon_tf, predict_btn], spacing=10),
            ft.Text("Predict FX Data", weight="bold", size=16),
            page.sim_header,
            page.sim_list_view,
            ft.Text("Final Results", weight="bold", size=16),
            page.final_results,
        ], expand=2, spacing=10),
    ], expand=True, spacing=20)

    # 🔳 Stack = main layout + floating logo on top-right
    return ft.Stack(
    expand=True,
    controls=[
        # Main page content
        ft.Container(
            expand=True,
            padding=20,
            content=main_layout,
        ),

        # Floating logo in top-right
        ft.Container(
            right=20,
            top=20,
            content=ft.GestureDetector(
                on_tap=lambda e: page.go("/"),
                mouse_cursor=ft.MouseCursor.CLICK,
                content=ft.Column(
                    controls=[
                        ft.Image(
                            src="Logo.png",
                            width=40,
                            height=40,
                            fit=ft.ImageFit.CONTAIN,
                        ),
                    ],
                    horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                )
            )
        ),

        # 🔹 Help Button in top-right (just left of the logo)
        ft.Container(
            right=80,
            top=20,
            content=ft.ElevatedButton(
                text="Help",
                icon=ft.Icons.HELP_OUTLINE,
                style=ft.ButtonStyle(
                    bgcolor=ft.Colors.BLUE_100,
                    color=ft.Colors.BLACK,
                    padding=ft.padding.symmetric(horizontal=12, vertical=8),
                    shape=ft.RoundedRectangleBorder(radius=8),
                ),
                on_click=lambda e: page.go("/help"),
            )
        ),
    ]
)



def on_fetch(page: ft.Page):
    pair, start, end = page.pair_dd.value, page.start_tf.value, page.end_tf.value
    try:
        sd = datetime.strptime(start, "%Y-%m-%d").date()
        ed = datetime.strptime(end,   "%Y-%m-%d").date()
        if sd > ed:
            raise ValueError
    except:
        page.snack_bar = ft.SnackBar(ft.Text("Invalid date format or range"))
        page.snack_bar.open = True
        page.update()
        return

    try:
        df = fetch_fx(pair, start, end)
    except Exception as err:
        page.snack_bar = ft.SnackBar(ft.Text(str(err)))
        page.snack_bar.open = True
        page.update()
        return

    page.img_chart.src_base64 = df_to_base64_plot(df, pair)
    full_idx = pd.date_range(sd, ed, freq="D")
    df_full  = df.reindex(full_idx)
    dates    = [d.date().isoformat() for d in full_idx]
    open_v   = df_full.iloc[:, 0]
    high_v   = df_full.iloc[:, 1]
    low_v    = df_full.iloc[:, 2]
    close_v  = df_full.iloc[:, 3]

    controls = []
    for i, d in enumerate(dates):
        fmt = lambda x: "N/A" if pd.isna(x) else f"{x:.4f}"
        controls.append(ft.Row([
            ft.Text(d,                     width=120),
            ft.Text(fmt(open_v.iloc[i]),  width=80, text_align="right"),
            ft.Text(fmt(high_v.iloc[i]),  width=80, text_align="right"),
            ft.Text(fmt(low_v.iloc[i]),   width=80, text_align="right"),
            ft.Text(fmt(close_v.iloc[i]), width=80, text_align="right"),
        ], spacing=5))
    page.list_view.controls = controls
    page.update()


def on_predict(page: ft.Page):
    try:
        h = int(page.horizon_tf.value)
        assert h > 0
    except:
        page.snack_bar = ft.SnackBar(ft.Text("Prediction days must be a positive integer"))
        page.snack_bar.open = True
        page.update()
        return

    # 调用主程序执行模拟，并获取 logs, final_caps, roi
    logs, final_caps, roi = cli_main(update_data=False, horizon=h)

    # 填充模拟结果表格
    sim_controls = []
    for entry in logs:
        day = entry['day']
        # USD/JPY
        sim_controls.append(ft.Row([
            ft.Text(str(day),               width=50),
            ft.Text("USD/JPY",            width=80),
            ft.Text(f"{entry['pred_USDJPY']:.4f}", width=80, text_align="right"),
            ft.Text(f"{entry['act_USDJPY']:.4f}",  width=80, text_align="right"),
            ft.Text(f"{entry['pnl_USDJPY']:.2f}",  width=80, text_align="right"),
            ft.Text(str(entry['pos_USDJPY']),      width=80, text_align="right"),
        ], spacing=5))
        # USD/EUR
        sim_controls.append(ft.Row([
            ft.Text(str(day),               width=50),
            ft.Text("USD/EUR",            width=80),
            ft.Text(f"{entry['pred_USDEUR']:.4f}", width=80, text_align="right"),
            ft.Text(f"{entry['act_USDEUR']:.4f}",  width=80, text_align="right"),
            ft.Text(f"{entry['pnl_USDEUR']:.2f}",  width=80, text_align="right"),
            ft.Text(str(entry['pos_USDEUR']),      width=80, text_align="right"),
        ], spacing=5))
        # USD/GBP
        sim_controls.append(ft.Row([
            ft.Text(str(day),               width=50),
            ft.Text("USD/GBP",            width=80),
            ft.Text(f"{entry['pred_USDGBP']:.4f}", width=80, text_align="right"),
            ft.Text(f"{entry['act_USDGBP']:.4f}",  width=80, text_align="right"),
            ft.Text(f"{entry['pnl_USDGBP']:.2f}",  width=80, text_align="right"),
            ft.Text(str(entry['pos_USDGBP']),      width=80, text_align="right"),
        ], spacing=5))
    page.sim_list_view.controls = sim_controls

    # 显示最终结果
    page.final_results.controls = [
        ft.Text(f"USD/JPY: {final_caps[0]:.2f}"),
        ft.Text(f"USD/EUR: {final_caps[1]:.2f}"),
        ft.Text(f"USD/GBP: {final_caps[2]:.2f}"),
        ft.Text(f"ROI: {roi:.4f}", color="blue", weight="bold"),
    ]
    page.update()

def main(page: ft.Page):
    def route_change(e):
        if page.route == "/":
            # Show landing page with background
            page.views.append(
                ft.View(
                    route="/",
                    controls=[landing_page(page)],
                    scroll=ft.ScrollMode.AUTO,
                )
            )

        elif page.route == "/simulator":
            # Don't show background here
            layout = fx_simulator_page(page)
            page.views.append(
                ft.View(
                    route="/simulator",
                    controls=[layout],
                )
            )
        
        elif page.route == "/help":
            # Don't show background here
            layout = help_page(page)
            page.views.append(
                ft.View(
                    route="/help",
                    controls=[layout],
                    scroll=ft.ScrollMode.AUTO,
                )
            )

        page.update()
    page.on_route_change = route_change
    page.go("/")


# 启动应用
if __name__ == "__main__":
    ft.app(target=main, view=ft.WEB_BROWSER, assets_dir="assets")
