# Chicago Transit Authority

The Chicago Transit Authority (CTA) is, as the name implies, the public transportation organization for the city of Chicago. From 2001 through May 2015, the CTA has averaged 825,586 bus rides per day and 458,871 train rides per day. This project attempts to supply you with tools for investigating the data that has been published.

## Installation

Download or clone this repository. You will need several python libraries: pandas, numpy, matplotlib, and pykml (if you think I am missing a dependency please let me know!).

## Examples

### Loading the data
The data is loaded into memory as a pandas dataframe. The columns describe the bus routes/train stops and the rows describe the date. A `NaN` value is used to indicate that the bus/train stop was not active on the given day.

```
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

```
bus.data.sum(axis=1).resample('AS').plot()
plt.hold(True)
train.data.sum(axis=1).resample('AS').plot()
(train.data.sum(axis=1).resample('AS') + bus.data.sum(axis=1).resample('AS')).plot()

plt.legend(['Bus', 'Train', 'Bus + Train'],loc='center left',bbox_to_anchor=(1,.5))
plt.ylim([0, plt.ylim()[1]])

```

The output should look something like

![](https://raw.githubusercontent.com/kbrose/dataViz/tree/master/Chicago/imgs/yearly_ridership.png)

### Geographic Routes
You can plot the routes in geographic coordinates (latitude/longitude (x,y) pairs), and additionally make the routes more opaque/more transparent based on average ridership:

```
# bus.routes() returns a list of all routes
bus.plot_route_shapes(bus.routes())
```

![](https://raw.githubusercontent.com/kbrose/dataViz/tree/master/Chicago/imgs/routes.png)

### Daily Ridership for a few routes
Daily ridership for different routes can be plotted easily as well:

```
# notice the flexible type and case-sensitivity for the route indicator
bus.plot_routes([2, '6', 28, 'x28', 'J14'],fillzero=True)
```

And after some zooming, it should look like

![](https://raw.githubusercontent.com/kbrose/dataViz/tree/master/Chicago/imgs/nov_dec_ridership.png)

### The Fast Fourier Transform
The Fast Fourier Transform (FFT) can be plotted for individual routes in a similar fashion:

```
bus.plot_fft(48)
```

![](https://raw.githubusercontent.com/kbrose/dataViz/tree/master/Chicago/imgs/fft.png)

(Note that 1/7 = 0.14285..., 2/7 = 0.2857..., and 3/7 = 0.4285...)

## Data Source(s)

Data for the CTA has been downloaded from [https://data.cityofchicago.org/](https://data.cityofchicago.org/). Additionally, some analyses may have been made using data collected using the CTA's API. You can request an API key from here: [bus](http://www.transitchicago.com/developers/bustracker.aspx), [train](http://www.transitchicago.com/developers/traintracker.aspx). All data used for any analyses is available in the `./data` sub-folder.
