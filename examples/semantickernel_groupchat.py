"""
The following code sample demonstrates an advanced agent group chat that utilizes 
SEO, SEM, CSR, and SDR Chat Completion Agents to collaborate on content creation, digital marketing and business development.
"""

import asyncio
import os

import azure.identity
from dotenv import load_dotenv
from openai import AsyncAzureOpenAI, AsyncOpenAI
from semantic_kernel import Kernel
from semantic_kernel.agents import AgentGroupChat, ChatCompletionAgent
from semantic_kernel.agents.strategies import (
    KernelFunctionSelectionStrategy,
    KernelFunctionTerminationStrategy,
)
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.contents import ChatHistoryTruncationReducer
from semantic_kernel.functions import KernelFunctionFromPrompt

# Define agent names
SEO_NAME = "SEO"
SEM_NAME = "SEM"
CSR_NAME = "CSR"
SDR_NAME = "SDR"

load_dotenv(override=True)
API_HOST = os.getenv("API_HOST", "github")


def create_kernel() -> Kernel:
    """Creates a Kernel instance with an Azure OpenAI ChatCompletion service."""
    kernel = Kernel()

    if API_HOST == "azure":
        token_provider = azure.identity.get_bearer_token_provider(azure.identity.DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default")
        chat_client = AsyncAzureOpenAI(
            api_version=os.environ["AZURE_OPENAI_VERSION"],
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            azure_ad_token_provider=token_provider,
        )
        chat_completion_service = OpenAIChatCompletion(ai_model_id=os.environ["AZURE_OPENAI_CHAT_MODEL"], async_client=chat_client)
    else:
        chat_client = AsyncOpenAI(api_key=os.environ["GITHUB_TOKEN"], base_url="https://models.inference.ai.azure.com")
        chat_completion_service = OpenAIChatCompletion(ai_model_id=os.getenv("GITHUB_MODEL", "gpt-4o"), async_client=chat_client)
    kernel.add_service(chat_completion_service)
    return kernel


async def main():
    # Create a single kernel instance for all agents.
    kernel = create_kernel()

    # Create ChatCompletionAgents using the same kernel.
    agent_seo = ChatCompletionAgent(
        kernel=kernel,
        name=SEO_NAME,
        instructions="""
Your responsibility is to optimize content for search engines.
Provide recommendations for keywords, meta descriptions, title tags, and overall content structure.
Ensure the content aligns with SEO best practices.

Conduct Comprehensive Keyword Research: Utilize tools like Google Keyword Planner, SEMrush, or Ahrefs to identify relevant keywords with high search volume and low competition.
Optimize Meta Tags: Craft unique and descriptive title tags and meta descriptions incorporating target keywords to improve click-through rates.
Structure Content with Header Tags: Use H1, H2, and H3 tags appropriately to organize content, enhancing readability and SEO.
Enhance Internal Linking: Create a logical internal linking structure to distribute page authority and improve site navigation.
Ensure Content Quality and Relevance: Develop high-quality, informative, and keyword-rich content that addresses user intent.
Build High-Quality Backlinks: Acquire backlinks from reputable and relevant websites through outreach, guest blogging, and content marketing.  
Engage in Social Media Marketing: Promote content on social media platforms to increase visibility and drive traffic to the website. 
Manage Online Reputation: Monitor and respond to online reviews and feedback to maintain a positive brand image.
Optimize for Local SEO: Claim and optimize local business listings, manage local citations, and encourage customer reviews to improve local search rankings.
Improve Website Loading Speed: Optimize images, leverage browser caching, and minimize code to enhance page load times.
Ensure Mobile-Friendliness: Implement responsive design to provide a seamless experience across all devices.
Implement Structured Data Markup: Use schema.org markup to help search engines understand the content and improve visibility in search results.  
Conduct Regular Technical Audits: Identify and fix crawl errors, broken links, and other technical issues that may hinder search engine indexing.
Secure the Website: Implement HTTPS to ensure secure connections, enhancing user trust and search engine rankings.
Develop a Content Calendar: Plan and schedule content creation to maintain consistency and address relevant topics.
Perform Content Audits: Regularly review existing content to update outdated information and improve quality.
Align Content with User Intent: Create content that addresses the specific needs and queries of the target audience.
Incorporate Multimedia Elements: Use images, videos, and infographics to enhance content engagement and SEO.
Monitor Website Performance Metrics: Use tools like Google Analytics to track traffic, bounce rates, and user behavior.
Track Keyword Rankings: Regularly assess the performance of targeted keywords to adjust strategies accordingly.
Generate SEO Reports: Provide regular reports detailing SEO performance, insights, and recommendations for improvement.
Analyze Competitor Strategies: Study competitors' SEO tactics to identify opportunities and threats.
Coordinate with Web Developers: Work closely with developers to implement technical SEO changes and resolve site issues.
Collaborate with Content Creators: Ensure content aligns with SEO strategies and maintains quality standards.
Stay Updated with SEO Trends: Continuously learn about changes in search engine algorithms and industry best practices to adapt strategies.
""",
    )

    agent_sem = ChatCompletionAgent(
        kernel=kernel,
        name=SEM_NAME,
        instructions="""
Your responsibility is to focus on search engine marketing (SEM) strategies.
Provide recommendations for paid advertising campaigns, bidding strategies, and ad copy optimization.
Ensure the content aligns with SEM goals and drives conversions.
Analyze current market trends and identify high-performing keywords to target.
Generate compelling and personalized ad copy tailored to specific audience segments.
Dynamically adjust keyword bids in real time to maximize ad placement and budget efficiency.
Monitor and analyze campaign performance, providing actionable insights and recommendations.
Segment and target audiences based on behavioral and demographic data for optimal reach.
Conduct competitor analysis to identify gaps, opportunities, and adapt strategies proactively.
Suggest and test landing page optimizations to improve conversion rates and align with ad messaging.
Track and manage SEM campaign costs, offering recommendations to optimize spending.
Integrate SEM strategies across multiple platforms to ensure consistent messaging and branding.
Continuously learn from campaign data to refine targeting, bidding, and creative strategies

""",
    )

    agent_csr = ChatCompletionAgent(
        kernel=kernel,
        name=CSR_NAME,
        instructions="""
Your responsibility is to ensure the content aligns with customer service and support needs.
Provide recommendations for addressing customer inquiries, providing helpful information, and improving customer satisfaction.
Ensure the content is customer-centric and easy to understand.
Respond promptly to customer inquiries via phone, email, chat, and social media platforms.​
Greet customers warmly and ascertain the reason for their inquiry.​
Listen actively to understand customer needs and concerns.​
Provide accurate information about products, services, and company policies.​
Maintain a positive, empathetic, and professional attitude toward customers at all times.​
Identify and assess customer issues to provide appropriate solutions.​
Troubleshoot technical problems and guide customers through step-by-step solutions.​
Escalate complex issues to higher-level support or management when necessary.​
Follow up with customers to ensure resolution and satisfaction.​
Process customer orders, returns, exchanges, and cancellations efficiently.​
Handle billing inquiries and process payments securely.​
Update customer accounts with accurate information.​
Monitor order status and provide updates to customers as needed.​
Identify opportunities to upsell or cross-sell products and services.​
Inform customers about promotions, discounts, and new offerings.​
Encourage customer loyalty by providing exceptional service and support.​
Collect and analyze customer feedback to improve service quality.​
Maintain detailed records of customer interactions, transactions, comments, and complaints.​
Prepare reports summarizing customer feedback and recurring issues.​
Document solutions and best practices for future reference.​
Ensure data privacy and compliance with company policies.​
Collaborate with team members and other departments to resolve customer issues.​
Participate in training sessions to stay updated on products and services.​
Provide constructive feedback to improve processes and customer experience.​
Adapt to new technologies and systems implemented by the company.​

""",
    )

    agent_sdr = ChatCompletionAgent(
        kernel=kernel,
        name=SDR_NAME,
        instructions="""
Your responsibility is to focus on sales development and lead generation.
Provide recommendations for crafting compelling sales messages, targeting the right audience, and driving lead conversions.
Ensure the content aligns with sales goals and generates qualified leads.
Respond promptly to customer inquiries via phone, email, chat, and social media platforms.​
Greet customers warmly and ascertain the reason for their inquiry.​
Listen actively to understand customer needs and concerns.​
Provide accurate information about products, services, and company policies.​
Maintain a positive, empathetic, and professional attitude toward customers at all times.​
Identify and assess customer issues to provide appropriate solutions.​
Troubleshoot technical problems and guide customers through step-by-step solutions.​
Escalate complex issues to higher-level support or management when necessary.​
Follow up with customers to ensure resolution and satisfaction.​
Process customer orders, returns, exchanges, and cancellations efficiently.​
Handle billing inquiries and process payments securely.​
Update customer accounts with accurate information.​
Monitor order status and provide updates to customers as needed.​
Identify opportunities to upsell or cross-sell products and services.​
Inform customers about promotions, discounts, and new offerings.​
Encourage customer loyalty by providing exceptional service and support.​
Collect and analyze customer feedback to improve service quality.​
Maintain detailed records of customer interactions, transactions, comments, and complaints.​
Prepare reports summarizing customer feedback and recurring issues.​
Document solutions and best practices for future reference.​
Ensure data privacy and compliance with company policies.​
Collaborate with team members and other departments to resolve customer issues.​
Participate in training sessions to stay updated on products and services.​
Provide constructive feedback to improve processes and customer experience.​
Adapt to new technologies and systems implemented by the company.​
""",
    )

    # Define a selection function to determine which agent should take the next turn.
    selection_function = KernelFunctionFromPrompt(
        function_name="selection",
        prompt=f"""
Examine the provided RESPONSE and choose the next participant.
State only the name of the chosen participant without explanation.
Never choose the participant named in the RESPONSE.

Choose only from these participants:
- {SEO_NAME}
- {SEM_NAME}
- {CSR_NAME}
- {SDR_NAME}

Rules:
- If RESPONSE is user input, it is {SEO_NAME}'s turn.
- If RESPONSE is by {SEO_NAME}, it is {SEM_NAME}'s turn.
- If RESPONSE is by {SEM_NAME}, it is {CSR_NAME}'s turn.
- If RESPONSE is by {CSR_NAME}, it is {SDR_NAME}'s turn.
- If RESPONSE is by {SDR_NAME}, it is {SEO_NAME}'s turn.

RESPONSE:
{{{{$lastmessage}}}}
""",
    )

    # Define a termination function where the SDR signals completion with "yes".
    termination_keyword = "yes"

    termination_function = KernelFunctionFromPrompt(
        function_name="termination",
        prompt=f"""
Examine the RESPONSE and determine whether the content has been deemed satisfactory.
If the content is satisfactory, respond with a single word without explanation: {termination_keyword}.
If specific suggestions are being provided, it is not satisfactory.
If no correction is suggested, it is satisfactory.

RESPONSE:
{{{{$lastmessage}}}}
""",
    )

    history_reducer = ChatHistoryTruncationReducer(target_count=5)

    # Create the AgentGroupChat with selection and termination strategies.
    chat = AgentGroupChat(
        agents=[agent_seo, agent_sem, agent_csr, agent_sdr],
        selection_strategy=KernelFunctionSelectionStrategy(
            initial_agent=agent_seo,
            function=selection_function,
            kernel=kernel,
            result_parser=lambda result: str(result.value[0]).strip() if result.value[0] is not None else SEO_NAME,
            history_variable_name="lastmessage",
            history_reducer=history_reducer,
        ),
        termination_strategy=KernelFunctionTerminationStrategy(
            agents=[agent_sdr],
            function=termination_function,
            kernel=kernel,
            result_parser=lambda result: termination_keyword in str(result.value[0]).lower(),
            history_variable_name="lastmessage",
            maximum_iterations=10,
            history_reducer=history_reducer,
        ),
    )

    print("Ready! Type your input, or 'exit' to quit, 'reset' to restart the conversation. " "You may pass in a file path using @<path_to_file>.")

    is_complete = False
    while not is_complete:
        print()
        user_input = input("User > ").strip()
        if not user_input:
            continue

        if user_input.lower() == "exit":
            is_complete = True
            break

        await chat.add_chat_message(message=user_input)
        try:
            async for response in chat.invoke():
                if response is None or not response.name:
                    continue
                print()
                print(f"# {response.name.upper()}:\n{response.content}")
        except Exception as e:
            print(f"Error during chat invocation: {e}")

        # Reset the chat's complete flag for the new conversation round.
        chat.is_complete = False


if __name__ == "__main__":
    asyncio.run(main())
