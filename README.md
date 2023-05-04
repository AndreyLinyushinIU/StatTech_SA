# Simulated Annealing used for Travelling Salesman problem optimization.

The attached .csv file has 1965 Russian cities locations, from which top 30 are chosen
by population. Since the coordinates are given in latitude and longitude, a Python
package ***haversine*** was used to measure the distance between cities.
The representation on 2D gif map below is just by using them as coordinates. The objective
here is the path length.

![](https://github.com/AndreyLinyushinIU/StatTech_SA/blob/main/annealing.gif)
To reproduce the gif use 
```
python3 gif_generate.py
```

![](https://github.com/AndreyLinyushinIU/StatTech_SA/blob/main/Comparison.png
)
To reproduce the plot use
```
python3 plot_generate.py
```