# Research Logbook -- Vattenfall turbine modelling

+++
---


```{admonition}  <span style = "color: blue; font-weight: 600; font-size: 25px">Collecting test data</span>
<span style = "color: blue; font-weight: 400; font-size: 20px">Both static and dynamic performance data are needed</span>
```


***
## Week 40 -- (Meeting on 2021-10-XX)
---

***
---

### Plan/action for Week 40
<span style = "font-weight: 400; font-size: 25px; color: red">
  1. Continue to preprocessing data <br />
  2. convert into Tensorflow dataset format for time series analysis <br />
  3. Looking for possiblies to apply other modelling techniques <br />
  4. Next week: probably at the end of October <br />
</span>

### Meeting minutes
* Week 44, Chalmers will get transient test data from Vattenfall. Currently, Vattenfall is busy in their testing for an EU project.
* Soon, Vattenfall will deliver some stationary BEP data to Chalmers for analysis
* Delay between pump_set and pump_speed due to control setting, but the ML methods cannot consider the delay time for the perfomrance modelling because the setting parameters from control mechanism are not happening to the actual turbine running system, but we can consider that when we implement own MPC control algorithms. Therefore, Vattenfall will try to deliver the data with slow control variation.
* Discharge_rate is a consequency of pump_speed and guide-open, blade_angle, tanker pressure, etc. One can model it, but should not use it as a feature to describe other parameters
* <span style = "font-weight: 400; font-size: 25px; color: blue">Clear objective: keep head_gross constant by adjust pump_speed, guide_open, blade_angle, and tank pressure.</span>
* <span style = "font-weight: 400; font-size: 25px; color: blue">Tasks: getting data from vattenfall for various stationary conditions to model turbine efficiency/BEP</span>
* Probably we need to talk to Berhanu in a late stage how to get the turbine efficiency from the data. Then, we can build BEP ML model to be compared with the conventional BEP figure.



### General update of research activities

* Preliminary investigation of the data from Berhalu@Vattenfall
* Very correlated and high frequency data for one scenario
* Promising to get the model for such prediction
