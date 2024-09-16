import geopandas as gpd
import pandas as pd

DATA_PATH = 'drive/MyDrive/current_research_projects/us_data/'

def get_yields(min_year, max_year, exclude_west):
    yields = pd.read_csv(DATA_PATH + 'yields/yields_1950-2022.csv').sort_values(['year', 'fips'])
    yields['fips'] = yields['fips'].apply(lambda x: str(x).zfill(5))
    yields_sample = yields[(yields['yield'] > 0) &
                           (yields['year'] <= max_year) & 
                           (yields['year'] >= min_year)]

    if exclude_west:
        counties = gpd.read_file(DATA_PATH + 'geo/counties_epsg4326.shp')

        def get_westernmost_point(x):
            try:
                coords = x.exterior.coords
            except:
                coords = []
                for poly in list(x.geoms):
                    coords += poly.exterior.coords
            return min([c[0] for c in list(coords)])

        counties['x_left'] = counties['geometry'].apply(get_westernmost_point)
        counties['fips'] = counties['GEOID'].apply(lambda x: str(x).zfill(5))
        counties_east = counties[counties['x_left'] > -100]
        return yields_sample[yields_sample['fips'].isin(counties_east['fips'])]

    else:
        return yields_sample


def get_sif():
    sif = pd.read_csv(DATA_PATH + 'sif/sif_county_mean_norm.csv').rename({'sif_mean': 'sif'}, axis=1)

    sif['date'] = pd.to_datetime(sif['date'])
    sif['year'] = sif['date'].dt.year
    sif['month'] = sif['date'].dt.month
    sif['fips'] = sif['fips'].apply(lambda x: str(x).zfill(5))
    return sif[(sif['month'] >= 3) & (sif['month'] <= 8)]


def get_modis_vi(vi):
    df = pd.read_csv(DATA_PATH + '%s/%s_county_mean_norm.csv' % (vi, vi)).drop_duplicates()
    df = df[['fips', 'date', 'year', 'normed_%s' % vi]].rename({'normed_%s' % vi: vi}, axis=1)

    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.month
    df['fips'] = df['fips'].apply(lambda x: str(x).zfill(5))

    df = df[(df['month'] >= 3) & (df['month'] <= 8)].sort_values('date')
    return df.dropna()


def get_max_vi(df, vi):
    return df.groupby(['fips', 'year'])[vi].max().reset_index().rename({vi: '%s_max' % vi}, axis=1)


def get_weather(min_year=2008):
    if min_year >= 2000:
        monthly_gdds = pd.read_csv(DATA_PATH + 'weather/ddays_2000_2020.csv')
    else:
        monthly_gdds = pd.read_stata(DATA_PATH +
            'weather/ddayByFipsAndMonth_areaCornWeighted_10station_year1950_2020.dta')
    monthly_gdds['fips'] = monthly_gdds['fips'].apply(lambda x: str(x).zfill(5))
    monthly_gdds = monthly_gdds[(monthly_gdds['month'] >= 3) & (monthly_gdds['month'] <= 8)]

    monthly_gdds['month'] += 0.5
    monthly_gdds['fips'] = monthly_gdds['fips'].apply(lambda x: str(x).zfill(5))

    # Make not cumulative by temp
    weather_mini = monthly_gdds[['fips', 'year', 'month', 'prec', 'dday29C']]\
        .assign(dday_10to29=monthly_gdds['dday10C'] - monthly_gdds['dday30C'])   # gdds/kdds only

    keep_cols = ['fips', 'year', 'month', 'tMin', 'tMax', 'tAvg2', 'prec', 'prec2']
    weather = monthly_gdds[keep_cols]  # All degree bands
    weather['dday_lt0'] = monthly_gdds['ddayMinus5C'] - monthly_gdds['dday0C']

    for i in range(3, 40, 3):
        weather['dday_%dto%d' % (i - 3, i)] = \
            monthly_gdds['dday%dC' % (i - 3)] - monthly_gdds['dday%dC' % i]

    weather['dday_gt39'] = monthly_gdds['dday39C']

    # Make cumulative by month
    def _make_cumulative(df, min_year=min_year):
        df = df[df['year'] >= min_year]
        dday_cols = [x for x in df.columns if x.startswith('dday')] + ['prec']
        cum_df = df[['fips', 'year', 'month'] + dday_cols].sort_values('month')
        for col in dday_cols:
            cum_df[col] = cum_df.groupby(['fips', 'year'])[col].cumsum()

        cum_df['prec2'] = cum_df['prec'] ** 2
        return cum_df

    cum_weather = _make_cumulative(weather, min_year)
    cum_weather_mini = _make_cumulative(weather_mini, min_year)

    return cum_weather, cum_weather_mini
