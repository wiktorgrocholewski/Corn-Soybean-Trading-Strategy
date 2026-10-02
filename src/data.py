# Data fetching, preprocessing and loading.
#
# Raw downloads live in data/raw/, derived datasets in data/processed/.
# Notebook 01 calls the fetch_* functions (optional, needs internet) and the
# build_* functions; every later notebook only calls the load_* functions.

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / 'data' / 'raw'
PROCESSED_DIR = ROOT / 'data' / 'processed'
CACHE_DIR = ROOT / 'data' / 'cache'
FIGURE_DIR = ROOT / 'report' / 'figures'
TABLE_DIR = ROOT / 'report' / 'tables'


#####################
# Universe definition
#####################

# Commodity ETFs (Yahoo tickers). ETFs roll their futures internally, so they
# avoid the artificial jumps of a continuous front-month futures series.
# Cotton: the iPath BAL ETN was delisted in 2023, so we use the London-listed
# WisdomTree Cotton ETC (COTN.L, quoted in USD).
ETF_TICKERS = {'corn': 'CORN', 'soybean': 'SOYB', 'wheat': 'WEAT', 'cotton': 'COTN.L'}

# Short names used for pairs / portfolios.
SHORT_NAMES = {'corn': 'CORN', 'soybean': 'SOYB', 'wheat': 'WEAT', 'cotton': 'COTN'}

# Front-month futures (longer history, but contaminated by roll jumps).
FUTURES_TICKERS = {'corn': 'ZC=F', 'soybean': 'ZS=F'}

# Weather locations (latitude, longitude), grouped by growing region.
LOCATIONS = {
    'us': {                                   # US Corn Belt (corn + soybean rotation)
        'central_iowa':     (42.0, -93.5),
        'central_illinois': (40.1, -89.4),
        'central_indiana':  (40.0, -86.2),
    },
    'brazil': {                               # Brazilian soy (and safrinha corn) regions
        'mato_grosso':    (-12.6, -55.5),
        'parana':         (-24.0, -51.5),
        'rio_grande_sul': (-29.5, -53.0),
    },
    'wheat': {                                # US wheat belt
        'kansas':       (38.5, -98.0),        # hard red winter wheat
        'oklahoma':     (35.5, -97.5),        # hard red winter wheat
        'north_dakota': (47.0, -100.0),       # spring wheat
    },
    'cotton': {                               # China / India cotton belts
        'korla':  (41.8, 86.1),
        'rajkot': (22.3, 70.8),
        # NOTE: data/raw/weather_cotton.csv was downloaded with longitude -79.1
        # (a point near Cuba). The coordinate below is the correct one; re-run
        # the download in notebook 01 to refresh the file.
        'nagpur': (21.1, 79.1),
    },
}

# Which weather regions are relevant for which commodity.
COMMODITY_REGIONS = {
    'corn':    ['us', 'brazil'],
    'soybean': ['us', 'brazil'],
    'wheat':   ['wheat'],
    'cotton':  ['cotton'],
}

# Agronomically relevant daily variables (crop stress).
WEATHER_VARIABLES = [
    'temperature_2m_mean',
    'temperature_2m_max',
    'temperature_2m_min',
    'precipitation_sum',
    'soil_moisture_0_to_7cm_mean',
    'soil_moisture_7_to_28cm_mean',
    'soil_moisture_28_to_100cm_mean',
    'soil_moisture_0_to_100cm_mean',
    'soil_temperature_0_to_7cm_mean',
    'et0_fao_evapotranspiration_sum',
    'relative_humidity_2m_mean',
    'vapour_pressure_deficit_max',
    'wet_bulb_temperature_2m_mean',
    'snowfall_water_equivalent_sum',
    'wind_speed_10m_max',
]


#####################
# 1. Downloading (optional: the raw files are already in data/raw/)
#####################

def fetch_yahoo(ticker, name, prefix='etf', start=None, end=None):
    """
    Download daily OHLCV data from Yahoo Finance and save it to
    data/raw/{prefix}_{name}.csv with columns close_{name}, high_{name}, ...
    """
    import yfinance as yf

    if start is None:
        data = yf.download(ticker, period='max', auto_adjust=True, timeout=10)
    else:
        data = yf.download(ticker, start=start, end=end, auto_adjust=True, timeout=10)

    # Flatten MultiIndex columns (e.g. ('Close', 'CORN') -> 'close')
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data.columns = [f'{col.lower()}_{name}' for col in data.columns]
    data.index.name = 'date'

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    data.to_csv(RAW_DIR / f'{prefix}_{name}.csv')
    print(f'{ticker:7s} -> {prefix}_{name}.csv  '
          f'({data.index.min().date()} to {data.index.max().date()}, {len(data)} rows)')
    return data


def fetch_weather(region, start_date='2005-01-01', end_date=None, variables=None):
    """
    Download daily weather for all locations of one region from the Open-Meteo
    archive API and save it to data/raw/weather_{region}.csv.

    Columns are named {variable}_{location}. Dates are normalised to midnight
    so that all locations align regardless of their time zone.
    """
    import openmeteo_requests
    import requests_cache
    from retry_requests import retry
    from datetime import date

    locations = LOCATIONS[region]
    variables = variables or WEATHER_VARIABLES
    end_date = end_date or date.today().isoformat()

    cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    client = openmeteo_requests.Client(session=retry_session)

    responses = client.weather_api(
        'https://archive-api.open-meteo.com/v1/archive',
        params={
            'latitude':  [lat for lat, _ in locations.values()],
            'longitude': [lon for _, lon in locations.values()],
            'start_date': start_date,
            'end_date': end_date,
            'daily': variables,
            'timezone': 'auto',
        },
    )

    frames = []
    for name, response in zip(locations, responses):
        print(f"  {name}: lat {response.Latitude():.2f}, lon {response.Longitude():.2f}")
        daily = response.Daily()
        dates = pd.date_range(
            start=pd.to_datetime(daily.Time(), unit='s', utc=True),
            end=pd.to_datetime(daily.TimeEnd(), unit='s', utc=True),
            freq=pd.Timedelta(seconds=daily.Interval()),
            inclusive='left',
        ).normalize().tz_localize(None)
        cols = {f'{var}_{name}': daily.Variables(j).ValuesAsNumpy()
                for j, var in enumerate(variables)}
        frames.append(pd.DataFrame(cols, index=pd.Index(dates, name='date')))

    weather = pd.concat(frames, axis=1)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    weather.to_csv(RAW_DIR / f'weather_{region}.csv')
    print(f'Saved weather_{region}.csv  {weather.shape}')
    return weather


#####################
# 2. Preprocessing
#####################

def rolling_weather(weather, window=30):
    """
    Backward-looking rolling aggregates of daily weather, using an
    agronomically sensible aggregation per variable:
      - precipitation, evapotranspiration, snowfall -> rolling SUM
      - soil moisture                               -> rolling MIN (worst-case stress)
      - max wind speed                              -> rolling MAX (storm damage)
      - everything else (temperature, humidity...)  -> rolling MEAN

    Output columns are named {column}_roll{window}d_{agg}.
    """
    def agg_method(col):
        if any(k in col for k in ['precipitation_sum', 'evapotranspiration_sum', 'snowfall']):
            return 'sum'
        if 'soil_moisture' in col:
            return 'min'
        if 'wind_speed_10m_max' in col:
            return 'max'
        return 'mean'

    rolled = {}
    for col in weather.columns:
        agg = agg_method(col)
        rolled[f'{col}_roll{window}d_{agg}'] = getattr(
            weather[col].rolling(window=window, min_periods=window), agg)()

    return pd.DataFrame(rolled, index=weather.index).iloc[window - 1:]


def build_full_dataset(assets=('corn', 'soybean', 'wheat', 'cotton'), window=30):
    """
    Build data/processed/full_dataset.csv: ETF close prices + rolling weather.

    Prices are OUTER-joined (US and London trading calendars differ), starting
    on the first day on which every asset trades. Use load_full_dataset(assets)
    to get the rows on which a given subset of assets has prices. Weather
    (calendar days) is forward-filled onto trading days, i.e. each trading day
    sees the most recent weather observation.
    """
    closes = [pd.read_csv(RAW_DIR / f'etf_{a}.csv', index_col='date',
                          parse_dates=True)[[f'close_{a}']] for a in assets]
    prices = pd.concat(closes, axis=1).sort_index()
    start = max(c.first_valid_index() for c in closes)
    prices = prices.loc[start:]

    weather = pd.concat(
        [pd.read_csv(RAW_DIR / f'weather_{r}.csv', index_col='date', parse_dates=True)
         for r in LOCATIONS], axis=1).sort_index()
    weather = rolling_weather(weather, window=window)
    weather = weather.reindex(prices.index, method='ffill')

    full = pd.concat([prices, weather], axis=1)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    full.to_csv(PROCESSED_DIR / 'full_dataset.csv')

    print(f'full_dataset.csv: {full.shape[0]} trading days '
          f'({full.index.min().date()} to {full.index.max().date()}), '
          f'{prices.shape[1]} price + {weather.shape[1]} weather columns')
    return full


def build_futures_dataset():
    """Build data/processed/futures_corn_soybean.csv (inner join of front-month closes)."""
    closes = [pd.read_csv(RAW_DIR / f'futures_{a}.csv', index_col='date',
                          parse_dates=True)[[f'close_{a}']] for a in FUTURES_TICKERS]
    futures = pd.concat(closes, axis=1, join='inner').dropna().sort_index()
    futures.columns = list(FUTURES_TICKERS)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    futures.to_csv(PROCESSED_DIR / 'futures_corn_soybean.csv')
    print(f'futures_corn_soybean.csv: {len(futures)} trading days '
          f'({futures.index.min().date()} to {futures.index.max().date()})')
    return futures


#####################
# 3. Loading (used by all analysis notebooks)
#####################

def load_full_dataset(assets=None):
    """
    Load data/processed/full_dataset.csv.

    assets : list of asset names (e.g. ['corn', 'soybean']). If given, keep
             only the trading days on which all of these assets have a price,
             and drop the price columns of the other assets.
    """
    df = pd.read_csv(PROCESSED_DIR / 'full_dataset.csv', index_col='date', parse_dates=True)
    if assets is not None:
        keep = [f'close_{a}' for a in assets]
        drop = [c for c in df.columns if c.startswith('close_') and c not in keep]
        df = df.drop(columns=drop).dropna(subset=keep)
    return df


def load_etf_prices(assets=('corn', 'soybean', 'wheat', 'cotton')):
    """ETF close prices on common trading days, columns renamed to short tickers."""
    df = load_full_dataset(list(assets))
    prices = df[[f'close_{a}' for a in assets]].copy()
    prices.columns = [SHORT_NAMES[a] for a in assets]
    return prices


def load_futures():
    """Front-month corn and soybean futures closes (cents per bushel)."""
    return pd.read_csv(PROCESSED_DIR / 'futures_corn_soybean.csv',
                       index_col='date', parse_dates=True)


def weather_columns(df, assets):
    """Rolling-weather columns of the regions relevant for the given assets."""
    locations = set()
    for a in assets:
        for region in COMMODITY_REGIONS[a]:
            locations |= set(LOCATIONS[region])
    return [c for c in df.columns
            if '_roll' in c and any(f'_{loc}_roll' in c for loc in locations)]


def train_test_split_date(index, train_frac=0.7):
    """First date of the test period for a chronological train/test split."""
    return index[int(len(index) * train_frac)]


def save_figure(fig, name):
    """Save a figure to report/figures/{name}.pdf (used by the LaTeX report)."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_DIR / f'{name}.pdf', bbox_inches='tight')


def save_table(df, name, float_format='%.4f'):
    """Save a results table to report/tables/{name}.csv (numbers quoted in the report)."""
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLE_DIR / f'{name}.csv', float_format=float_format)
