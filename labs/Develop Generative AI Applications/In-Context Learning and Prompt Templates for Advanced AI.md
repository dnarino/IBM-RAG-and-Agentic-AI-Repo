<p style="text-align:center">
    <a href="https://skills.network" target="_blank">
    <img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/assets/logos/SN_web_lightmode.png" width="200" alt="Skills Network Logo"  />
    </a>
</p>


# Master Prompt Engineering and LangChain PromptTemplates


Estimated time needed: **45** minutes


## Overview


You're stepping into the world of prompt engineering, where each command you craft has the power to guide intelligent LLM systems toward specific outcomes. In this tutorial, you will explore the foundational aspects of prompt engineering, dive into advanced in-context learning techniques such as few-shot and self-consistent learning, and learn how to effectively use tools like LangChain to create prompt templates.

You'll start by learning the basics—how to formulate prompts that communicate effectively with AI. From there, you'll explore how LangChain prompt templates can simplify and enhance this process, making it more structured and efficient.

As you progress, you'll learn to apply these skills in practical scenarios, creating sophisticated applications like QA bots and text summarization tools. By using LangChain prompt templates, you'll see firsthand how structured prompting can streamline the development of these applications, transforming complex requirements into clear, concise tasks for AI.


<img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/ai8G4tOU4mksEYfv5wsghA/prompt%20engineering.png" width="50%" alt="indexing"/>


By the end of this tutorial, you'll not only master various prompt engineering techniques but also gain hands-on experience applying these techniques to real-world problems, ensuring you're well-prepared to harness the full potential of AI in diverse settings.


## __Table of Contents__

<ol>
    <li><a href="#Objectives">Objectives</a></li>
    <li>
        <a href="#Setup">Setup</a>
        <ol>
            <li><a href="#Install-required-libraries">Install required libraries</a></li>
            <li><a href="#Import-required-libraries">Import required libraries</a></li>
            <li><a href="#Set-up-the-LLM">Set up the LLM</a></li>
        </ol>
    </li>
    <li>
        <a href="#Prompt-engineering">Prompt engineering</a>
        <ol>
            <li>
                <a href="#Basic-prompt">Basic prompt</a>
                <ol>
                    <li><a href="#Exercise-1">Exercise 1</a></li>
                </ol>
            </li>
            <li>
                <a href="#Zero-shot-prompt">Zero-shot prompt</a>
                <ol>
                    <li><a href="#Exercise-2">Exercise 2</a></li>
                </ol>
            </li>
            <li>
                <a href="#One-shot-prompt">One-shot prompt</a>
                <ol>
                    <li><a href="#Exercise-3">Exercise 3</a></li>
                </ol>
            </li>
            <li><a href="#Few-shot-prompt">Few-shot prompt</a></li>
            <li>
                <a href="#Chain-of-thought-(CoT)-prompt">Chain-of-thought (CoT) prompt</a>
                <ol>
                    <li><a href="#Exercise-4">Exercise 4</a></li>
                </ol>
            </li>
            <li><a href="#Self-consistency">Self-consistency</a></li>
        </ol>
    </li>
    <li>
        <a href="#Applications-of-prompting-in-different-use-cases">Applications of prompting in different use cases</a>
        <ol>
            <li><a href="#Introduction-to-LangChain">Introduction to LangChain</a></li>
            <li><a href="#Prompt-template">Prompt template</a></li>
            <li><a href="#Text-summarization">Text summarization</a></li>
            <li><a href="#Question-answering">Question answering</a></li>
            <li><a href="#Text-classification">Text classification</a></li>
            <li><a href="#Code-generation">Code generation</a></li>
            <li><a href="#Role-playing">Role playing</a></li>
            <li><a href="#Exercise-5">Exercise 5</a></li>
        </ol>
    </li>
    <li><a href="#Conclusion">Conclusion</a></li>
    <li><a href="#Authors">Authors</a></li>
</ol>


## Objectives

After completing this lab, you will be able to:

- **Understand the basics of prompt engineering**: Gain a solid foundation in how to effectively communicate with LLM using prompts, setting the stage for more advanced techniques.

- **Master advanced prompt techniques**: Learn and apply advanced prompt engineering methods such as few-shot and self-consistent learning to optimize the LLM's response.

- **Utilize LangChain prompt templates**: Become proficient in using LangChain's prompt template to structure and optimize your interactions with LLMs.

- **Develop practical LLM agents**: Acquire the skills to create and implement agents such as QA bots and text summarization tools using the LangChain prompt template, translating theoretical knowledge into practical solutions.


----


## Setup


For this lab, you will use the following libraries:

*   [`ibm-watsonx-ai`](https://ibm.github.io/watson-machine-learning-sdk/index.html): Enables the use of LLMs from IBM's watsonx.ai.
*   [`langchain`](https://www.langchain.com/): Provides various chain and prompt functions from LangChain.
*   [`langchain-ibm`](https://python.langchain.com/v0.1/docs/integrations/llms/ibm_watsonx/): Facilitates integration between LangChain and IBM watsonx.ai.


### Install required libraries

The following required libraries are __not__ preinstalled in the Skills Network Labs environment. __You must run the following cell__ to install them:

**Note:** The version has been pinned to ensure compatibility. It's recommended that you do the same. While future updates may be available, the pinned version ensures that the library continues to support this lab.

This might take approximately 1-2 minutes.

`%%capture` has been used to capture the installation, so you won’t see the process. However, once the installation is complete, a number will appear next to the cell.



```python
%%capture
!pip install "ibm-watsonx-ai==1.0.8" --user
!pip install "langchain==0.2.11" --user
!pip install "langchain-ibm==0.1.7" --user
!pip install "langchain-core==0.2.43" --user
```

After you install the libraries, restart your kernel:



```python
import os
os._exit(00)
```

**ATTENTION**: if the code above doesn't work, you can restart the kernal manuallu by clicking the **Restart the kernel** icon as shown in the following screenshot:


<img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/6DtO5_X9SAK4tYJCnsre2w/restart-kernel.png" width="80%" alt="Restart kernel">


Once the kernel has been restarted, move on to the next part `Import required libraries`.


### Import required libraries


_It is recommended that you import all required libraries in one place (here):_



```python
# You can also use this section to suppress warnings generated by your code:
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings('ignore')

# IBM WatsonX imports
from ibm_watsonx_ai.foundation_models import Model
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes

from langchain_ibm import WatsonxLLM
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableSequence
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.chains import LLMChain  # Still using this for backward compatibility
```

### Set up the LLM


In this section, you will build an LLM using IBM watsonx.ai. The following code initializes a Granite model on IBM's watsonx.ai platform and wraps it into a function that allows for repeat use.


Some key parameters are explained here:
- `model_id` specifies which model you want to use. There are various model options available; refer to the [Foundation Models](https://ibm.github.io/watsonx-ai-python-sdk/foundation_models.html) documentation for more options. In this tutorial, you'll use the `ibm/granite-4-h-small` model.
- `parameters` define the model's configuration. Set five commonly used parameters for this tutorial. To explore additional commonly used parameters, you can run the code `GenParams().get_example_values()`. If no custom parameters are passed to the function, the model will use `default_params`.
- `credentials` and `project_id` are required to successfully run LLMs from watsonx.ai. (Leave `credentials` and `project_id` as they are so you don't need to create your own keys to run models.) This ensures you can run the model inside this lab environment. However, if you want to run the model locally, refer to this [tutorial](https://medium.com/the-power-of-ai/ibm-watsonx-ai-the-interface-and-api-e8e1c7227358) for creating your own keys.
- `WatsonxLLM()` is used to create an instance of the LLM.


## API Disclaimer
This lab uses LLMs provided by **Watsonx.ai**. This environment has been configured to allow LLM use without API keys so you can prompt them for **free (with limitations)**. With that in mind, if you wish to run this notebook **locally outside** of Skills Network's JupyterLab environment, you will have to **configure your own API keys**. Please note that using your own API keys means that you will incur personal charges.

### Running Locally
If you are running this lab locally, you will need to configure your own API keys. This lab uses the `WatsonxLLM` module from `IBM`, we'll need to initialize the `granite_llm` with credentials to use the LLM locally. Fill out the commented out `api_key` field in the cell below and pass in the credentials if you are running locally.


##### Run the following code to initialize the LLM locally. If you are using Skills Network's JupyterLab environment, skip the following cell code.



```python
def llm_model(prompt_txt, params=None):
    
    model_id = "ibm/granite-4-h-small"

    default_params = {
        "max_new_tokens": 256,
        "min_new_tokens": 0,
        "temperature": 0.5,
        "top_p": 0.2,
        "top_k": 1
    }

    if params:
        default_params.update(params)

    # Set up credentials for WatsonxLLM
    url = "https://us-south.ml.cloud.ibm.com"
    api_key = "your api key here"
    project_id = "skills-network"

    credentials = {
        "url": url,
        # "api_key": api_key
        # uncomment the field above and replace the api_key with your actual Watsonx API key
    }
    
    # Create LLM directly
    granite_llm = WatsonxLLM(
        model_id=model_id,
        credentials=credentials,
        project_id=project_id,
        params=default_params
    )
    
    response = granite_llm.invoke(prompt_txt)
    return response
```

##### Run the following code to initialize the LLM using Skills Network's JupyterLab environment



```python
def llm_model(prompt_txt, params=None):
    
    model_id = "ibm/granite-4-h-small"

    default_params = {
        "max_new_tokens": 256,
        "min_new_tokens": 0,
        "temperature": 0.5,
        "top_p": 0.2,
        "top_k": 1
    }

    url = "https://us-south.ml.cloud.ibm.com"
    project_id = "skills-network"
    
    granite_llm = WatsonxLLM(
        model_id=model_id,
        project_id=project_id,
        url=url,
        params=default_params
    )
    
    response = granite_llm.invoke(prompt_txt)
    return response

```

Let's run the following code to see some other commonly used parameters and their default values.



```python
GenParams().get_example_values()
```

## Prompt engineering

[Prompt engineering](https://www.ibm.com/think/topics/prompt-engineering?utm_source=skills_network&utm_content=in_lab_content_link&utm_id=Lab-In-Context+Learning+and+Prompt+Templates-v3-GenAIcourse_1741386184) is the art and science of crafting effective inputs for large language models to generate desired outputs. As language models have evolved in capability and size, so too has the importance of how we communicate with them. Prompt engineering involves strategically designing text prompts that guide an AI model's responses toward specific goals, formats, or reasoning patterns.

**[In-context learning](https://research.ibm.com/blog/demystifying-in-context-learning-in-large-language-model?utm_source=skills_network&utm_content=in_lab_content_link&utm_id=Lab-In-Context+Learning+and+Prompt+Templates-v3-GenAIcourse_1741386184)** represents one of the most fascinating capabilities of modern large language models. It is a model's ability to "learn" from examples provided directly within the prompt itself, without any updates to its underlying parameters or weights. This capability allows models to adapt to new tasks or domains simply by demonstrating what success looks like through examples.

By combining the principles of prompt engineering with the power of in-context learning, developers can guide language models to perform a remarkably diverse range of tasks with unprecedented flexibility and efficiency. Let's now explore these methods in detail.


### Basic prompt

A **basic prompt** is the simplest form of prompting, where you provide a short text or phrase to the model without any special formatting or instructions. The model generates a continuation based on patterns it has learned during training. Basic prompts are useful for exploring the model's capabilities and understanding how it naturally responds to minimal input.


In this example, let's introduce a basic prompt that uses specific parameters to guide the language model's response. You'll then define a simple prompt and retrieve the model's response.

The prompt used is "The wind is." Let the model generate itself.



```python
params = {
    "max_new_tokens": 128,
    "min_new_tokens": 10,
    "temperature": 0.5,
    "top_p": 0.2,
    "top_k": 1
}

prompt = "The wind is "

# Getting a reponse from the model with the provided prompt and new parameters
response = llm_model(prompt, params)
print(f"prompt: {prompt}\n")
print(f"response : {response}\n")
```

As you can see from the response, the model continues generating content based on the initial prompt, "The wind is." You might notice that the response appears truncated or incomplete. This is because you have set the `max_new_tokens,` which restricts the number of tokens the model can generate.

Try adjusting the parameters and observe how the response changes.


### Exercise 1
Experiment with different basic prompts by changing the input phrase. Try these prompts and compare the different responses:
1. "The future of artificial intelligence is"
2. "Once upon a time in a distant galaxy"
3. "The benefits of sustainable energy include"



```python
## Your code here

```

<details>
    <summary>Click here for hints</summary>

```python
params = {
    "max_new_tokens": 128, # Try 256 or 512 for more detailed answers
    "min_new_tokens": 10, # Increase to 25-50 if you want more substantial answers
    "temperature": 0.5, # Controls randomness in generation (0.0-1.0)
                       # Lower (0.1-0.3): More focused, consistent, factual responses
                      # Higher (0.7-1.0): More creative, diverse, unpredictable outputs
    "top_p": 0.2, # Nucleus sampling - considers only highest probability tokens
                       # Lower values (0.1-0.3): More conservative, focused text
                       # Higher values (0.7-0.9): More diverse vocabulary and ideas
    "top_k": 1 # Limits token selection to top k most likely tokens
                       # 1 = greedy decoding (always picks most likely token)
                       # Try 40-50 for more varied outputs
}

# Compare responses to different prompts
prompts = [
    "The future of artificial intelligence is",
    "Once upon a time in a distant galaxy",
    "The benefits of sustainable energy include"
]

for prompt in prompts:
    response = llm_model(prompt, params)
    print(f"prompt: {prompt}\n")
    print(f"response : {response}\n")
```
</details>


### Zero-shot prompt

[**Zero-shot prompting**](https://www.ibm.com/think/topics/zero-shot-prompting?utm_source=skills_network&utm_content=in_lab_content_link&utm_id=Lab-In-Context+Learning+and+Prompt+Templates-v3-GenAIcourse_1741386184) is a technique where the model performs a task without any examples or prior specific training on that task. This approach tests the model's ability to understand instructions and apply its knowledge to a new context without demonstration. Zero-shot prompts typically include clear instructions about what the model should do, allowing it to leverage its pre-trained knowledge effectively.


---
Here is an example of a zero-shot prompt:

Zero-shot learning is crucial for testing a model's ability to apply its pre-trained knowledge to new, unseen tasks without additional training. This capability is valuable for gauging the model's generalization skills.

In this example, let's demonstrate a zero-shot learning scenario using a prompt that asks the model to classify a statement without any prior specific training on similar tasks. The prompt requests the model to assess the truthfulness of the statement: "The Eiffel Tower is located in Berlin." After defining the prompt, you'll execute it with default parameters and print the response.

This approach helps you understand how well the model can handle direct questions based on its underlying knowledge and reasoning abilities.


Try running the prompt to see the model's capacity to correctly analyze and respond to factual inaccuracies.



```python
prompt = """Classify the following statement as true or false: 
            'The Eiffel Tower is located in Berlin.'

            Answer:
"""
response = llm_model(prompt, params)
print(f"prompt: {prompt}\n")
print(f"response : {response}\n")
```

The model responds with the 'False' answer, which is correct. It also gives the reason for the answer.


### Exercise 2
Create zero-shot prompts for the following tasks:
1. Write a prompt that asks the model to classify a movie review as positive or negative.
2. Create a prompt that instructs the model to summarize a paragraph about climate change.
3. Design a prompt that asks the model to translate an English phrase to Spanish without showing any examples.



```python
## Starter code: provide your solutions in the TODO parts

# 1. Prompt for Movie Review Classification
movie_review_prompt = ## TODO

# 2. Prompt for Climate Change Paragraph Summarization
climate_change_prompt = ## TODO

# 3. Prompt for English to Spanish Translation
translation_prompt = ## TODO

responses = {}
responses["movie_review"] = ## TODO
responses["climate_change"] = ## TODO
responses["translation"] = ## TODO

for prompt_type, response in responses.items():
    print(f"=== {prompt_type.upper()} RESPONSE ===")
    print(response)
    print()
```

<details>
    <summary>Click here for hints</summary>

```python
# 1. Prompt for Movie Review Classification
movie_review_prompt = """
Classify the following movie review as either 'positive' or 'negative'.

Review: "I was extremely disappointed by this film. The plot was predictable, the acting was wooden, and the special effects looked cheap. I can't recommend this to anyone."

Classification:
"""

# 2. Prompt for Climate Change Paragraph Summarization
climate_change_prompt = """
Summarize the following paragraph about climate change in no more than two sentences.

Paragraph: "Climate change refers to long-term shifts in temperatures and weather patterns. These shifts may be natural, but since the 1800s, human activities have been the main driver of climate change, primarily due to the burning of fossil fuels like coal, oil and gas, which produces heat-trapping gases. The consequences of climate change include more frequent and severe droughts, storms, and heat waves, rising sea levels, melting glaciers, and warming oceans which can directly impact biodiversity, agriculture, and human health."

Summary:
"""

# 3. Prompt for English to Spanish Translation
translation_prompt = """
Translate the following English phrase into Spanish.

English: "I would like to order a coffee with milk and two sugars, please."

Spanish:
"""

responses = {}
responses["movie_review"] = llm_model(movie_review_prompt)
responses["climate_change"] = llm_model(climate_change_prompt)
responses["translation"] = llm_model(translation_prompt)

for prompt_type, response in responses.items():
    print(f"=== {prompt_type.upper()} RESPONSE ===")
    print(response)
    print()
```
</details>


### One-shot prompt

[**One-shot prompting**](https://www.ibm.com/think/topics/one-shot-prompting?utm_source=skills_network&utm_content=in_lab_content_link&utm_id=Lab-In-Context+Learning+and+Prompt+Templates-v3-GenAIcourse_1741386184) provides the model with a **single** example of the task before asking it to perform a similar task. This technique gives the model a pattern to follow, improving its understanding of the desired output format and style. One-shot learning is particularly useful when you want to guide the model's response without extensive examples.


Here is a one-shot learning example where the model is given a single example to help guide its translation from English to French.

The prompt provides a sample translation pairing, "How is the weather today?" translated to "Comment est le temps aujourd'hui?" This example serves as a guide for the model to understand the task context and desired format. The model is then tasked with translating a new sentence, "Where is the nearest supermarket?" without further guidance.



```python
params = {
    "max_new_tokens": 20,
    "temperature": 0.1,
}

prompt = """Here is an example of translating a sentence from English to French:

            English: “How is the weather today?”
            French: “Comment est le temps aujourd'hui?”
            
            Now, translate the following sentence from English to French:
            
            English: “Where is the nearest supermarket?”
            
"""
response = llm_model(prompt, params)
print(f"prompt: {prompt}\n")
print(f"response : {response}\n")
```

The model's response shows how it applies the structure and context provided by the initial example to translate the new sentence.


Consider experimenting with different sentences or adjusting the parameters to see how these changes impact the model's translations.


### Exercise 3
Develop one-shot prompts for these scenarios:
1. Create a prompt with one example of a formal email, then ask the model to write another formal email on a different topic.
2. Provide one example of converting a technical concept into a simple explanation, then ask the model to explain a different concept.
3. Give one example of extracting keywords from a sentence, then ask the model to extract keywords from a new sentence.



```python
## Starter code: provide your solutions in the TODO parts

# 1. One-shot prompt for formal email writing
formal_email_prompt = ## TODO

# 2. One-shot prompt for simplifying technical concepts
technical_concept_prompt = ## TODO

# 3. One-shot prompt for keyword extraction
keyword_extraction_prompt = ## TODO

responses = {}
responses["formal_email"] = ## TODO
responses["technical_concept"] = ## TODO
responses["keyword_extraction"] = ## TODO

for prompt_type, response in responses.items():
    print(f"=== {prompt_type.upper()} RESPONSE ===")
    print(response)
    print()
```

<details>
    <summary>Click here for hints</summary>

```python
# 1. One-shot prompt for formal email writing
formal_email_prompt = """
Here is an example of a formal email requesting information:

Subject: Inquiry Regarding Product Specifications for Model XYZ-100

Dear Customer Support Team,

I hope this email finds you well. I am writing to request detailed specifications for your product Model XYZ-100. Specifically, I am interested in learning about its dimensions, power requirements, and compatibility with third-party accessories.

Could you please provide this information at your earliest convenience? Additionally, I would appreciate any available documentation or user manuals that you could share.

Thank you for your assistance in this matter.

Sincerely,
John Smith

---

Now, please write a formal email to a university admissions office requesting information about their application deadline and required documents for the Master's program in Computer Science:

"""

# 2. One-shot prompt for simplifying technical concepts
technical_concept_prompt = """
Here is an example of explaining a technical concept in simple terms:

Technical Concept: Blockchain
Simple Explanation: A blockchain is like a digital notebook that many people have copies of. When someone writes a new entry in this notebook, everyone's copy gets updated. Once something is written, it can't be erased or changed, and everyone can see who wrote what. This makes it useful for recording important information that needs to be secure and trusted by everyone.

---

Now, please explain the following technical concept in simple terms:

Technical Concept: Machine Learning
Simple Explanation:
"""

# 3. One-shot prompt for keyword extraction
keyword_extraction_prompt = """
Here is an example of extracting keywords from a sentence:

Sentence: "Cloud computing offers businesses flexibility, scalability, and cost-efficiency for their IT infrastructure needs."
Keywords: cloud computing, flexibility, scalability, cost-efficiency, IT infrastructure

---

Now, please extract the main keywords from the following sentence:

Sentence: "Sustainable agriculture practices focus on biodiversity, soil health, water conservation, and reducing chemical inputs."
Keywords:
"""

responses = {}
responses["formal_email"] = llm_model(formal_email_prompt)
responses["technical_concept"] = llm_model(technical_concept_prompt)
responses["keyword_extraction"] = llm_model(keyword_extraction_prompt)

for prompt_type, response in responses.items():
    print(f"=== {prompt_type.upper()} RESPONSE ===")
    print(response)
    print()
```
</details>


### Few-shot prompt

[**Few-shot prompting**](https://www.ibm.com/think/topics/few-shot-prompting?utm_source=skills_network&utm_content=in_lab_content_link&utm_id=Lab-In-Context+Learning+and+Prompt+Templates-v3-GenAIcourse_1741386184) extends the one-shot approach by providing multiple examples (typically 2-5) before asking the model to perform the task. These examples establish a clearer pattern and context, helping the model better understand the expected output format, style, and reasoning. This technique is particularly effective for complex tasks where a single example might not convey all the nuances.


Here is an example of few-shot learning by classifying emotions from text statements. 

Let's provide the model with three examples, each labeled with an appropriate emotion—joy, frustration, and sadness—to establish a pattern or guideline on how to categorize emotions in statements.

After presenting these examples, let's challenge the model with a new statement: "That movie was so scary I had to cover my eyes." The task for the model is to classify the emotion expressed in this new statement based on the learning from the provided examples. 



```python
#parameters: Set `max_new_tokens` to 10, which constrains the model to generate brief responses

params = {
    "max_new_tokens": 10,
}

prompt = """Here are few examples of classifying emotions in statements:

            Statement: 'I just won my first marathon!'
            Emotion: Joy
            
            Statement: 'I can't believe I lost my keys again.'
            Emotion: Frustration
            
            Statement: 'My best friend is moving to another country.'
            Emotion: Sadness
            
            Now, classify the emotion in the following statement:
            Statement: 'That movie was so scary I had to cover my eyes.’
            

"""
response = llm_model(prompt, params)
print(f"prompt: {prompt}\n")
print(f"response : {response}\n")
```

The parameters are set with `max_new_tokens` to 10, which constrains the model to generate brief responses, focusing on the essential output without elaboration.


The model's response demonstrates its ability to use the provided few examples to understand and classify the emotion of the new statement effectively following the same pattern in examples.


### Chain-of-thought (CoT) prompt

[**Chain-of-thought (CoT) prompting**](https://www.ibm.com/think/topics/chain-of-thoughts?utm_source=skills_network&utm_content=in_lab_content_link&utm_id=Lab-In-Context+Learning+and+Prompt+Templates-v3-GenAIcourse_1741386184) encourages the model to break down complex problems into step-by-step reasoning before arriving at a final answer. By explicitly showing or requesting intermediate steps, this technique improves the model's problem-solving abilities and reduces errors in tasks requiring multi-step reasoning. CoT is particularly effective for mathematical problems, logical reasoning, and complex decision-making tasks.


Here is an example of the CoT prompting technique, designed to guide the model through a sequence of reasoning steps to solve a problem. In this example, the problem is a simple arithmetic question: “A store had 22 apples. They sold 15 apples today and received a new delivery of 8 apples. How many apples are there now?”

The CoT technique involves structuring the prompt by instructing the model to “Break down each step of your calculation.” This encourages the model to include explicit reasoning steps, mimicking human-like problem-solving processes.



```python
params = {
    "max_new_tokens": 512,
    "temperature": 0.5,
}

prompt = """Consider the problem: 'A store had 22 apples. They sold 15 apples today and got a new delivery of 8 apples. 
            How many apples are there now?’

            Break down each step of your calculation

"""
response = llm_model(prompt, params)
print(f"prompt: {prompt}\n")
print(f"response : {response}\n")
```

From the response of the model, you can see the prompt directs the model to:

1. Add the initial number of apples to the apples received in the new delivery.
2. Subtract the number of apples sold from the sum obtained in the first step.


By breaking down the problem into specific steps, the model is better able to understand the sequence of operations required to arrive at the correct answer.


### Exercise 4
Create CoT prompts for these scenarios:
1. Write a prompt that asks the model to think through whether a student should study tonight or go to a movie with friends, considering their upcoming test in two days.
2. Write a prompt that instructs the model to explain the step-by-step process of making a peanut butter and jelly sandwich.



```python
## Starter code: provide your solutions in the TODO parts

# 1. Prompt for decision-making process
decision_making_prompt = ## TODO

# 2. Prompt for explaining a process
sandwich_making_prompt = ## TODO

responses = {}
responses["decision_making"] = ## TODO
responses["sandwich_making"] = ## TODO

for prompt_type, response in responses.items():
    print(f"=== {prompt_type.upper()} RESPONSE ===")
    print(response)
    print()
```

<details>
    <summary>Click here for hints</summary>

```python
# 1. Prompt for decision-making process
decision_making_prompt = """
Consider this situation: A student is trying to decide whether to study tonight or go to a movie with friends. They have a test in two days.

Think through this decision step-by-step, considering the pros and cons of each option, and what factors might be most important in making this choice.
"""

# 2. Prompt for explaining a process
sandwich_making_prompt = """
Explain how to make a peanut butter and jelly sandwich.

Break down each step of the process in detail, from gathering ingredients to finishing the sandwich.
"""

responses = {}
responses["decision_making"] = llm_model(decision_making_prompt)
responses["sandwich_making"] = llm_model(sandwich_making_prompt)

for prompt_type, response in responses.items():
    print(f"=== {prompt_type.upper()} RESPONSE ===")
    print(response)
    print()
```
</details>


### Self-consistency

[**Self-consistency**](https://www.promptingguide.ai/techniques/consistency) is an advanced technique in which the model generates multiple independent solutions or answers to the same problem, then evaluates these different approaches to determine the most consistent or reliable result. This method enhances accuracy by leveraging the model's ability to approach problems from different angles and identify the most robust solution through comparison and verification.


This example demonstrates the self-consistency technique by reasoning through multiple calculations for a single problem. The problem posed is: “When I was 6, my sister was half my age. Now I am 70, what age is my sister?”

The prompt instructs, “Provide three independent calculations and explanations, then determine the most consistent result.” This encourages the model to engage in critical thinking and consistency checking, both of which are vital for complex decision-making processes.



```python
params = {
    "max_new_tokens": 512,
}

prompt = """When I was 6, my sister was half of my age. Now I am 70, what age is my sister?

            Provide three independent calculations and explanations, then determine the most consistent result.

"""
response = llm_model(prompt, params)
print(f"prompt: {prompt}\n")
print(f"response : {response}\n")
```

The model's response demonstrates three different calculations and explanations, each using a distinct logical approach to determine the sister's age.

Self-consistency can help identify the most accurate and reliable answer in scenarios where multiple plausible solutions exist.


## Applications of prompting in different use cases


In this section, we'll demonstrate how to leverage LangChain's prompt templates to build practical applications with consistent, reproducible results. Each application follows a common pattern using the LCEL approach:

1. Define the content or problem to be addressed.
2. Create a template with variables for dynamic content.
3. Convert the template into a LangChain PromptTemplate.
4. Build a chain using the pipe operator `|` to connect:

    - Input variables
    - The prompt template
    - The LLM
    - An output parser


5. Invoke the chain with specific inputs to generate results.

This structured approach enables you to create reusable components for various NLP tasks while maintaining flexibility to adjust parameters and inputs. You'll see how this pattern applies across different use cases.


### Introduction to LangChain 

[LangChain](https://www.langchain.com/) is a powerful framework designed to simplify the development of applications powered by language models. Built to address the challenges of working with LLMs in practical settings, LangChain provides a standardized interface for connecting models with various data sources and application environments.

LangChain serves as an abstraction layer, making it easier to build complex LLM applications without handling the low-level details of model interaction. This framework has become a standard tool in the LLM ecosystem, supporting a wide range of use cases from chatbots to document analysis systems.

In this section, we'll focus on LangChain's prompt template capabilities, demonstrating how they can be used to create structured, reproducible interactions with language models across different application types."


### Prompt template


[Prompt templates](https://python.langchain.com/v0.2/docs/concepts/#prompt-templates) are a key concept in LangChain. They help translate user input and parameters into instructions for a language model. These templates can be used to guide a model's response, helping it understand the context and generate relevant and coherent language-based outputs.

A prompt template acts as a reusable structure for generating prompts with dynamic values. It allows you to define a consistent format while leaving placeholders for variables that change with each use case. This approach makes prompting more systematic and maintainable, especially when working with complex applications.

**Modern LangChain (as of 2025) offers two main approaches to working with templates:**

- The traditional `LLMChain` approach
- The newer LangChain Expression Language (LCEL) pattern using the pipe operator `|` for more flexible composition

LCEL has become the recommended pattern for building LangChain applications as it offers better composability, clearer visualization of data flow, and more flexibility when constructing complex chains.

**To use a prompt template with LCEL, you typically follow these steps:**

- Define your template with variables in curly braces `{}`
- Create a `PromptTemplate` instance
- Build a chain using the pipe operator `|` to connect components
- Invoke the chain with your input values

Let's initialize an LLM first, then demonstrate this approach. In this section, we will use the model `meta-llama/llama-3-3-70b-instruct`:



```python
model_id = "ibm/granite-4-h-small"

parameters = {
    GenParams.MAX_NEW_TOKENS: 256,  # this controls the maximum number of tokens in the generated output
    GenParams.TEMPERATURE: 0.5, # this randomness or creativity of the model's responses
}

url = "https://us-south.ml.cloud.ibm.com"
project_id = "skills-network"

llm = WatsonxLLM(
        model_id=model_id,
        url=url,
        project_id=project_id,
        params=parameters
    )
llm
```

Use the `PromptTemplate` to create a template for a string-based prompt. In this template, you'll define two parameters: `adjective` and `content`. These parameters allow for the reuse of the prompt across different situations. For instance, to adapt the prompt to various contexts, simply pass the relevant values to these parameters.



```python
template = """Tell me a {adjective} joke about {content}.
"""
prompt = PromptTemplate.from_template(template)
prompt 
```

Now, let's take a look at how the prompt has been formatted.



```python
prompt.format(adjective="funny", content="chickens")
```

From the response, you can see that the prompt is formatted according to the specified context.


To ensure consistent formatting of the prompts, we will define a helper function `format_prompt`. This function takes a dictionary of variables and applies them to our prompt template. It ensures that all placeholder variables (like {adjective} and {content}) are properly replaced with their values before the prompt is sent to the language model.



```python
from langchain_core.runnables import RunnableLambda

# Define a function to ensure proper formatting
def format_prompt(variables):
    return prompt.format(**variables)
```

The following code builds a chain using the LCEL (LangChain Expression Language) pattern. This chain connects components using the pipe operator (`|`) to create a processing flow. The chain takes input variables, passes them through the prompt template, sends the formatted prompt to the LLM, and uses a string output parser to return the final response.



```python
# Create the chain with explicit formatting
joke_chain = (
    RunnableLambda(format_prompt)
    | llm 
    | StrOutputParser()
)

# Run the chain
response = joke_chain.invoke({"adjective": "funny", "content": "chickens"})
print(response)
```

From the response, you can see the LLM came up with a funny joke about chickens.

To use this prompt in another context, simply replace the variables accordingly.



```python
response = joke_chain.invoke({"adjective": "sad", "content": "fish"})
print(response)
```

In the following sections, you will learn how to create agents capable of completing various tasks using prompt templates.


### Text summarization


Here is a text summarization agent designed to help summarize the content you provide to the LLM. The LCEL chain takes your content as input, processes it through the prompt template, sends it to the language model, and returns a concise summary.

You can store the content to be summarized in a variable, allowing for repeated use with different texts:



```python
content = """
    The rapid advancement of technology in the 21st century has transformed various industries, including healthcare, education, and transportation. 
    Innovations such as artificial intelligence, machine learning, and the Internet of Things have revolutionized how we approach everyday tasks and complex problems. 
    For instance, AI-powered diagnostic tools are improving the accuracy and speed of medical diagnoses, while smart transportation systems are making cities more efficient and reducing traffic congestion. 
    Moreover, online learning platforms are making education more accessible to people around the world, breaking down geographical and financial barriers. 
    These technological developments are not only enhancing productivity but also contributing to a more interconnected and informed society.
"""

template = """Summarize the {content} in one sentence.
"""
prompt = PromptTemplate.from_template(template)

# Create the LCEL chain
summarize_chain = (
    RunnableLambda(format_prompt)
    | llm 
    | StrOutputParser()
)

# Run the chain
summary = summarize_chain.invoke({"content": content})
print(summary)
```

### Question answering


Here is a Q&A agent built using the LCEL pattern.

This agent enables the LLM to learn from the provided content and answer questions based on what it has learned. Occasionally, if the LLM does not have sufficient information, it may generate a speculative answer. To manage this, we'll specifically instruct it to respond with "Unsure about the answer" if it is uncertain about the correct response.

The chain takes both the content (context) and question as inputs, processing them through our template before sending them to the LLM:



```python
content = """
    The solar system consists of the Sun, eight planets, their moons, dwarf planets, and smaller objects like asteroids and comets. 
    The inner planets—Mercury, Venus, Earth, and Mars—are rocky and solid. 
    The outer planets—Jupiter, Saturn, Uranus, and Neptune—are much larger and gaseous.
"""

question = "Which planets in the solar system are rocky and solid?"

template = """
    Answer the {question} based on the {content}.
    Respond "Unsure about answer" if not sure about the answer.
    
    Answer:
    
"""
prompt = PromptTemplate.from_template(template)

# Create the LCEL chain
qa_chain = (
    RunnableLambda(format_prompt)
    | llm 
    | StrOutputParser()
)

# Run the chain
answer = qa_chain.invoke({"question": question, "content": content})
print(answer)
```

### Text classification


Here is a text classification agent designed to categorize text into predefined categories. This example employs zero-shot learning, where the agent classifies text without prior exposure to related examples.

Using the LCEL approach, we create a chain that takes both the text to be classified and the available categories as inputs:



```python
text = """
    The concert last night was an exhilarating experience with outstanding performances by all artists.
"""

categories = "Entertainment, Food and Dining, Technology, Literature, Music."

template = """
    Classify the {text} into one of the {categories}.
    
    Category:
    
"""
prompt = PromptTemplate.from_template(template)

# Create the LCEL chain
classification_chain = (
    RunnableLambda(format_prompt)
    | llm 
    | StrOutputParser()
)

# Run the chain
category = classification_chain.invoke({"text": text, "categories": categories})
print(category)
```

### Code generation


Here is an example of an SQL code generation agent built with LCEL. This agent is designed to generate SQL queries based on provided descriptions. It interprets the requirements from your input and translates them into executable SQL code.

The chain takes your natural language description and transforms it into a properly formatted SQL query:



```python
description = """
    Retrieve the names and email addresses of all customers from the 'customers' table who have made a purchase in the last 30 days. 
    The table 'purchases' contains a column 'purchase_date'
"""

template = """
    Generate an SQL query based on the {description}
    
    SQL Query:
    
"""
prompt = PromptTemplate.from_template(template)

# Create the LCEL chain
sql_generation_chain = (
    RunnableLambda(format_prompt) 
    | llm 
    | StrOutputParser()
)

# Run the chain
sql_query = sql_generation_chain.invoke({"description": description})
print(sql_query)
```

### Role playing


You can also configure the LLM to assume specific roles as defined by us, enabling it to follow predetermined rules and behave like a task-oriented chatbot.

This approach separates the role definition from the prompt structure, allowing for easy role-switching without rewriting the entire prompt. The key components are:

- `role`: Specifies the character, expertise, or persona the LLM should embody
- `tone`: Defines the communication style and emotional quality of responses
- `question`: Contains the user's query that needs addressing

By parameterizing these elements, you can rapidly change the LLM's behavior by adjusting single variables rather than rewriting entire prompts. This pattern is particularly valuable for building conversational agents that need to serve different functions or adapt to various contexts.

For example, the code below configures the LLM to act as a game master. In this role, the LLM answers questions about games while maintaining an engaging and immersive tone, enhancing the user experience. You can test the bot by asking questions related to tabletop role-playing games or game mastering. Try asking about game rules, storytelling techniques, player management, or setting descriptions such as:

1. "Who are you?"
2. "What are the basic rules of Dungeons & Dragons?"
3. "How do I create a balanced encounter for my players?"
4. "Can you describe a mysterious forest setting for my adventure?"
5. "What's a good puzzle I could use in my dungeon?"
6. "How do I handle a player who is constantly interrupting others?"

The function is written within a while loop, allowing continuous interaction. **To exit the loop and terminate the conversation, type "quit," "exit," or "bye" into the input box.**



```python
role = """
    Dungeon & Dragons game master
"""

tone = "engaging and immersive"

template = """
    You are an expert {role}. I have this question {question}. I would like our conversation to be {tone}.
    
    Answer:
    
"""
prompt = PromptTemplate.from_template(template)

# Create the LCEL chain
roleplay_chain = (
    RunnableLambda(format_prompt)
    | llm 
    | StrOutputParser()
)

# Create an interactive chat loop
while True:
    query = input("Question: ")
    
    if query.lower() in ["quit", "exit", "bye"]:
        print("Answer: Goodbye!")
        break
        
    response = roleplay_chain.invoke({"role": role, "question": query, "tone": tone})
    print("Answer: ", response)
```

### Exercise 5

**Create an LCEL Chain with Custom Formatting**

In this exercise, you'll create your own LCEL chain that uses prompt templates to build a custom application.

**Task:** Create a product review analyzer that can:
1. Identify the sentiment (positive, negative, or neutral).
2. Extract mentioned product features.
3. Provide a one-sentence summary of the review.

**Steps:**
1. Create a prompt template with placeholders for the review text.
2. Build an LCEL chain that formats your prompt properly.
3. Process the sample reviews and display the results.
4. Try modifying the chain to change the output format.

**Sample input:**
```python
reviews = [
    "I love this smartphone! The camera quality is exceptional and the battery lasts all day. The only downside is that it heats up a bit during gaming.",
    "This laptop is terrible. It's slow, crashes frequently, and the keyboard stopped working after just two months. Customer service was unhelpful."
]



```python
## Starter code: provide your solutions in the TODO parts
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# First initialize your LLM
model_id = "ibm/granite-4-h-small" ## Or you can use other LLMs available via watsonx.ai

# Use these parameters
parameters = {
    GenParams.MAX_NEW_TOKENS: 512,  # this controls the maximum number of tokens in the generated output
    GenParams.TEMPERATURE: 0.2, # this randomness or creativity of the model's responses
}

url = "https://us-south.ml.cloud.ibm.com"
project_id = "skills-network"

# TODO: Initialize your LLM
llm = ## TODO

# Here is an example template you can use
template = """
Analyze the following product review:
"{review}"

Provide your analysis in the following format:
- Sentiment: (positive, negative, or neutral)
- Key Features Mentioned: (list the product features mentioned)
- Summary: (one-sentence summary)
"""

# TODO: Create your prompt template
product_review_prompt = ## TODO

# TODO: Create a formatting function
def format_review_prompt(variables):
    # Your code here
    pass

# TODO: Build your LCEL chain
review_analysis_chain = (
    # Your code here
)

# Example reviews to process
reviews = [
    "I love this smartphone! The camera quality is exceptional and the battery lasts all day. The only downside is that it heats up a bit during gaming.",
    "This laptop is terrible. It's slow, crashes frequently, and the keyboard stopped working after just two months. Customer service was unhelpful."
]

# TODO: Process the reviews
for review in reviews:
    # Your code here
    pass
```

<details>
    <summary>Click here for hints</summary>

```python
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser

model_id = "ibm/granite-4-h-small"

parameters = {
    GenParams.MAX_NEW_TOKENS: 512,  # this controls the maximum number of tokens in the generated output
    GenParams.TEMPERATURE: 0.2, # this randomness or creativity of the model's responses
}

url = "https://us-south.ml.cloud.ibm.com"
project_id = "skills-network"

llm = WatsonxLLM(
        model_id=model_id,
        url=url,
        project_id=project_id,
        params=parameters
    )

# Create the prompt template
template = """
Analyze the following product review:
"{review}"

Provide your analysis in the following format:
- Sentiment: (positive, negative, or neutral)
- Key Features Mentioned: (list the product features mentioned)
- Summary: (one-sentence summary)
"""

product_review_prompt = PromptTemplate.from_template(template)

# Create a formatting function
def format_review_prompt(variables):
    return product_review_prompt.format(**variables)

# Build the LCEL chain
review_analysis_chain = (
    RunnableLambda(format_review_prompt)
    | llm 
    | StrOutputParser()
)

# Process the reviews
reviews = [
    "I love this smartphone! The camera quality is exceptional and the battery lasts all day. The only downside is that it heats up a bit during gaming.",
    "This laptop is terrible. It's slow, crashes frequently, and the keyboard stopped working after just two months. Customer service was unhelpful."
]

for i, review in enumerate(reviews):
    print(f"==== Review #{i+1} ====")
    result = review_analysis_chain.invoke({"review": review})
    print(result)
    print()
```
</details>


## Conclusion

Congratulations on completing this lab on prompt engineering and LangChain prompt templates! You've successfully navigated from basic prompting techniques to more advanced approaches including zero-shot, one-shot, few-shot learning, as well as chain-of-thought reasoning and self-consistency prompting. You then applied these concepts practically using LangChain's prompt templates with the modern LCEL pattern to create various applications.

By learning how to properly structure prompts and build composable chains with the pipe operator (`|`), you've gained essential skills for developing robust LLM applications. These techniques provide a solid foundation for creating more complex systems and getting the most out of any language model you work with.


## Authors


[Hailey Quach](https://www.haileyq.com) is a Data Scientist at IBM.

[Kang Wang](https://author.skills.network/instructors/kang_wang) is a Data Scientist at IBM. He is also a PhD Candidate in the University of Waterloo.

[Faranak Heidari](https://author.skills.network/instructors/faranak_heidari) is a Data Scientist at IBM. 


<!---
## Change log

|Date (YYYY-MM-DD)|Version|Changed By|Change Description|
|-|-|-|-|
|2025-03-07|1.1|Hailey Quach|Updated lab|
|2025-04-03|1.2|Jojy John|ID Reviewed|
|2025-04-03|1.2|Rahul Rawat|QA pass|
--->


© Copyright IBM Corporation. All rights reserved.

