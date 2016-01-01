# Chicago Transit Authority

The Chicago Transit Authority (CTA) is, as the name implies, the public transportation organization for the city of Chicago. From 2001 through May 2015, the CTA has averaged 825,586 bus rides per day and 458,871 train rides per day. This project attempts to supply you with tools for investigating the data that has been published.

Most of the focus thus far has been on tools for investigating bus ridership. The train ridership is a more complex beast; we get train ridership by the *station*, not by the train. If a station serves more than one train, then we cannot know for certain how many people used each of the different lines at the station. This makes any analysis much more complex.

## Installation

Download or clone this repository. You will need several python libraries: pandas, numpy, matplotlib, and optionally pykml to do geographic processes (if you think I am missing a dependency please let me know!).

## Examples

It is my hope that the documentation in the main code file (`./src/cta.py`) will be good enough so that you can start to use the library right away. The few examples here are more to showcase what kinds of investigation this project attempts to facilitate.

### Loading the data
The data is loaded into memory as a pandas dataframe. The columns describe the bus routes/train stops and the rows describe the date. A `NaN` value is used to indicate that the bus/train stop was not active on the given day.

```python
import cta

bus = cta.bus()
train = cta.train()

# bus dataframe
d1 = bus.data

# train dataframe
d2 = train.data
```

### Yearly Ridership
You can easily see what the yearly trends of the ridership for buses, trains, and buses and trains combined:

```python
bus.data.sum(axis=1).resample('AS').plot()
plt.hold(True)
train.data.sum(axis=1).resample('AS').plot()
(train.data.sum(axis=1).resample('AS') + bus.data.sum(axis=1).resample('AS')).plot()

plt.legend(['Bus', 'Train', 'Bus + Train'],loc='center left',bbox_to_anchor=(1,.5))
plt.ylim([0, plt.ylim()[1]])

```

The output should look something like

![](https://raw.githubusercontent.com/kbrose/chicago-data/master/imgs/yearly_ridership.png)

### Geographic Routes
You can plot the routes in geographic coordinates (latitude/longitude (x,y) pairs), and additionally make the routes more opaque/more transparent based on average ridership:

```python
# bus.routes() returns a list of all routes
bus.plot_route_shapes(bus.routes())
```

![](https://raw.githubusercontent.com/kbrose/chicago-data/master/imgs/routes.png)

### Daily Ridership for a few routes
Daily ridership for different routes can be plotted easily as well:

```python
# notice the flexible type and case-sensitivity for the route indicator
bus.plot_routes([2, '6', 28, 'x28', 'J14'],fillzero=True)
```

We can zoom in on just a couple months:

![](https://raw.githubusercontent.com/kbrose/chicago-data/master/imgs/nov_dec_ridership.png)

### The Fast Fourier Transform
The Fast Fourier Transform ([FFT](https://en.wikipedia.org/wiki/Fast_Fourier_transform)) can be plotted for individual routes in a similar fashion:

```python
bus.plot_fft(48)
```

which results in

![](https://raw.githubusercontent.com/kbrose/chicago-data/master/imgs/fft.png)

(Note that 1/7 = 0.14285..., 2/7 = 0.2857..., and 3/7 = 0.4285....)

The FFT is normalized so that it sums to 1. The hope is that this allows for meaningful comparison between the FFT of different routes, so that a route with higher overall variance would not dominate over a route with smaller variance.

### Distance Between Routes
The distance between two bus routes is defined as the shortest distance between any two bus-stops along each of the routes. For a given route, the list of all routes that are within some specified distance of `d` meters can be found. To see what routes arise, we can plot those routes that are within the distance.

```python
within_dist = bus.routes_within_dist(d=50,routes=146,accurate=False)
bus.plot_shapes(bus.routes_within_dist(50, 146, False)['146'])
```

You can specify multiple routes, or none (which will default to finding all routes within the specified distance for all other routes). The `accurate` flag effects how distance is computed. If `accurate` is set to false, then distance will be computed as the typical (Euclidean) distance on the [Mercator Projection](https://en.wikipedia.org/wiki/Mercator_projection) (where the projection has been centered on Chicago), otherwise it will use a more complicated, but more accurate, formula (see [here](http://www.movable-type.co.uk/scripts/latlong.html) for the distance formula given in terms of the Earth's radius, and [here](https://en.wikipedia.org/wiki/Earth_radius) for the equation of the Earth's radius at a given latitude). It appears as if setting `accurate` to False results in a speed-up of about 4X, but does have noticeable changes in results for some routes.

## Data Source(s)

Data for the CTA has been downloaded from [https://data.cityofchicago.org/](https://data.cityofchicago.org/). Additionally, some analyses may have been made using data collected using the CTA's API. You can request an API key from here: [bus](http://www.transitchicago.com/developers/bustracker.aspx), [train](http://www.transitchicago.com/developers/traintracker.aspx). All data used for any analyses is available in the `./data` sub-folder.
