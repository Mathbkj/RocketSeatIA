import os;
import json;
import yfinance as yf;
from crewai import Task, Agent,Crew;

from langchain.tools import Tool;
from langchain_openai import ChatOpenAI;
from langchain_community.tools import DuckDuckGoSearchResults;
from crewai.process import Process;
import streamlit as st;




def ticket__fetcher(ticket):
  stock = yf.download(ticket,"2023-08-09","2024-08-09")
  return stock

finance__tool = Tool(
  name="Yahoo Finance Tool",
  func=lambda ticket:ticket__fetcher(ticket),
  description="Fetches financial data for a given ticket."
)

os.environ["OPENAI_API_KEY"] = st.secrets['OPENAI_API_KEY']
llm = ChatOpenAI(model="gpt-3.5-turbo")

analystAgent = Agent(
  role="Senior Stock Price Analyst",
  goal="Find the ticket and analyze trends",
  backstory="You're highly experienced in financial data analysis therefore being able to provide good feedback over them",
  verbose=True,
  llm=llm,
  max_iter=5,
  memory=True,
  tools=[finance__tool],
  allow_delegation=False
)
getStockPrice=Task(
  description="Analyze the stock {ticket} price history and create a trend analysis of up, down or sideways.",
  expected_output=""""Specify the current trend stock price -up, down or sideways.
  eg. stock='AAPL,price UP'""""",
  agent=analystAgent,
)
search_tool = DuckDuckGoSearchResults(backend='news',max_results=10)

newsAnalyst = Agent(
  role="Stock News Analyst",
  goal="""Create a short summary of the market news related to the stock {ticket} company.
  Specify the current trend -up, down or sideways with the news context.
  For each stock asset, specify a number between 0 and 100, where 0 is extreme fear and 100 is extreme greed.""",
  backstory="""You're highly experienced in analyzing the market trends and news and have tracked assets for over 10 years.
  You're also master level analyst in the traditional market and have deep understanding of human psychology.
  You understand news, their titles and information, but you look at those with a healthy dose of skepticism as well as considering the news source.""",
  verbose=True,
  llm=llm,
  max_iter=5,
  memory=True,
  tools=[search_tool],
  allow_delegation=False
)
get_news = Task(
  description="""Take the stock and always include BTC to it(if not requested)
  Use the search tool to search each one individually.
  
  Compose the results into a helpful report""",
  expected_output="""
  A summary of the overall market and one sentence summary for each request asset
  Include a fear/greed score for each asset based on the news. Use format:
  <STOCK ASSET>
  <SUMMARY BASED ON NEWS>
  <TREND PREDICTION>
  <FEAR/GREED SCORE>
""",
agent=newsAnalyst
)
stockAnalystReport = Agent(
  role="Senior Stocks Analyst Writer",
  goal="""Analyse trends price and news and write an insightful compelling and informative
  3 paragraph long newsletter based on the stock report and price trend""",
  backstory="""You're widely accepted as the best stock analyst in the market.
  You understand complex concepts and create compelling stories
  and narratives that resonate with wider audiences
  
  You understand macro factors and combine multiple theories - eg. cycle theory and fundamental analysis
  You're able to hold multiple opinions when analyzing data""",
  verbose=True,
  llm=llm,
  max_iter=5,
  memory=True,
  allow_delegation=True,
)
writeAnalysis=Task(
  description="""Use the stock trend and the stock news report
  to create an analysis and write a newsletter about the {ticket} company that is brief and highlights
  the most important points.
  Focus on the stock price trend, news and fear/greed score. What are the near future considerations?
  Include the previous analyses of stock trend and news summary.""",
  expected_output="""
An eloquent 3 paragraphs newsletter formatted as markdown in an easy readable manner. It should contain:
- 3 bullets
- Introduction=>set the overall picture and spike interest
- Main Part=>Provides the meat of the analysis including the news summary and fear/greed scores
- Summary=>key facts and concrete future predictions-up,down or sideways.
""",
agent=stockAnalystReport,
context=[getStockPrice,get_news]
)
crew = Crew(
  agents=[analystAgent,newsAnalyst,stockAnalystReport],
  tasks=[getStockPrice,get_news,writeAnalysis],
  process=Process.hierarchical,
  full_output=True,
  share_crew=False,
  manager_llm=llm,
  verbose=True,
  max_iter=15,
)


with st.sidebar:
  st.header("Enter the Stock to Research")
  with st.form(key='research_form'):
    topic=st.text_input("Select the Ticket")
    submit_button = st.form_submit_button(label="Run Research")
  if(submit_button):
    if not topic:st.error("Please fill the ticket field")
  else:
    results = crew.kickoff(inputs={'ticket':topic})
    st.subheader("Results of your research:")
    st.write(results['final__output'])




  

