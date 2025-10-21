# Flight checker 👨🏻‍✈️

This agent is designed to tell you where a flight is currently located by its call sign/flight number.  
Via a Yacana Tool, it calls an external HTTP API called OpenSkyNet to get flight information.  

## Prerequisites

### Packages
You must install those packages first:

```shell
pip install yacana numpy requests
```

### Environment variables

You must set your OpenSkyNet token as an environment variable.
To get an OpenSkyNet token, follow [this short tutorial](https://openskynetwork.github.io/opensky-api/rest.html).

Finally, use the below to set your token:
```bash
export TOKEN_OPENSKY={your_token_value}
```

### Defining the LLM to use

In `agent.py`, you can define which LLM to use by changing the model_name from the Agent class.

## Usage
Then, you can kickstart a conversation with an agent by:

```
python3 ./agent.py
```

Example
```
prompt> Where is the flight with call sign VS20?

agent> The current location of flight VS20 is over the United Kingdom. The specific coordinates are approximately 55.2154 latitude and -1.7301 longitude.
```

# Thanks
## OpenSkyNet
Matthias Schäfer, Martin Strohmeier, Vincent Lenders, Ivan Martinovic and Matthias Wilhelm.
"Bringing Up OpenSky: A Large-scale ADS-B Sensor Network for Research".
In Proceedings of the 13th IEEE/ACM International Symposium on Information Processing in Sensor Networks (IPSN), pages 83-94, April 2014.
