from yacana import OllamaAgent, Tool, Task, OllamaModelSettings
from flight_tool import get_flight_data_by_callsign


ms = OllamaModelSettings(temperature=0.7)

agent = OllamaAgent("Flight assistant", "llama3.1:8b", system_prompt=
"""
You are a flight specialist. 
- Only respond to questions about flights or aviation; politely refuse all others.
- You ARE ABLE to provide real-time flight information via specialized tools we will give you.
- Only call a tool if the user provides a valid flight number, call sign, or explicitly requests real-time flight information.
- For general aviation questions, respond directly using your knowledge.
- If asked about a specific flight, always include the country or ocean/sea over which the flight is currently located, if available.
""", model_settings=ms)

# No more examples
flight_t: Tool = Tool("Flight_Tool",
"""
Gives real-time coordinates (latitude, longitude), returns its status amongst other detailed.
You ARE able to get real-time information.
The data is real-time and calls the opensky-network API. 
What's cool is that you only need the flight number or flight 
call sign to get flight data from that method.
Before calling this tool, ask yourself if you got a call sign or a flight number. 
If you don't have a flight call sign or a flight number,
there's no valid reason to call this tool.
""", get_flight_data_by_callsign, optional=True, usage_examples=[{"flight_number": "AA101"},
                                                                {"flight_number": "BA234"},
                                                                {"flight_number": "VS20"},
                                                                {"flight_number": "UA4567"},
                                                                {"flight_number": "QR7890"},
                                                                {"flight_number": "EM1201"},
                                                                {"flight_number": "CX3214"}])

while True:
    prompt = input(">")

    Task(prompt, agent, tools=[flight_t]).solve()

    result = Task("""Now, re-write about that result into a comprehensive summary that's comprehensible for a customer
    that doesn't care about how you work technically, but that only
    wants info about their flight or aviation in general.
    Remember that you're now talking to the customer, so don't bother writing an introductory paragraph.
    Go straight to the point.
    """, agent, forget=True).solve().content
    print(result)
