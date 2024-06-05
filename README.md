## Build an advanced RAG router based assistant with Amazon Bedrock

This repository contains the code samples that will let participants learn how to use the [Retrieval Augmented Generation (RAG)](https://docs.aws.amazon.com/sagemaker/latest/dg/jumpstart-foundation-models-customize-rag.html) router based architecture with [Amazon Bedrock](https://aws.amazon.com/bedrock/) and [Amazon OpenSearch Serverless (AOSS)](https://aws.amazon.com/opensearch-service/features/serverless/) to quickly build a secure assistant that uses the most up-to-date information to converse with users. Participants will also learn how this assistant will use dialog-guided information retrieval to respond to users.

### Overview

[Amazon Bedrock](https://aws.amazon.com/bedrock/) is a fully managed service that offers a choice of high-performing Foundation Models (FMs) from leading AI companies accessible through a single API, along with a broad set of capabilities you need to build generative AI applications, simplifying development while maintaining privacy and security.

[Large Language Models (LLMs)](https://en.wikipedia.org/wiki/Large_language_model) are a type of Foundation Model that can take natural langauge as input, with the ability to process and understand it, and produce natural language as the output. LLMs can also can perform tasks like classification, summarization, simplification, entity recognition, etc.

LLMs are usually trained offline with data that is available until that point of time. As a result, LLMs will not have knowledge of the world after that date. Additionally, LLMs are trained on very general domain corpora, making them less effective for domain-specific tasks. And then, LLMs have the tendency to hallucinate where the model generates text that is incorrect, nonsensical, or not real. Using a [Retrieval Augment Generation (RAG)](https://docs.aws.amazon.com/sagemaker/latest/dg/jumpstart-foundation-models-customize-rag.html) mechanism can help mitigate all these issues. A RAG architecture involves retrieving data that closely matches the text in the user's prompt, from an external datasource, and using it to augment the prompt before sending to the LLM. This prompt augmentation will provide the context that the LLM can use to respond to the prompt.

When there are mulitple datasources, there is a need to route the retreival request to the appropriate datasource before performing the actual retrieval. This is a RAG router pattern.

This repository contains code that will walk you through the process of building an advanced RAG router based assistant using a Large Language Model (LLM) hosted on [Amazon Bedrock](https://aws.amazon.com/bedrock/) and using [Knowledge Bases for Amazon Bedrock](https://docs.aws.amazon.com/bedrock/latest/userguide/knowledge-base.html) for vectorizing, storing, and retrieving data through semantic search. [Amazon OpenSearch Serverless](https://aws.amazon.com/opensearch-service/features/serverless/) will be used as the vector index.

### To get started

1. Choose an AWS Account to use and make sure to create all resources in that Account.
2. Identify an AWS Region that has [Amazon Bedrock with Anthropic Claude 3 and Titan Embeddings G1 - Text](https://docs.aws.amazon.com/bedrock/latest/userguide/models-regions.html) models.
3. In that Region, create a new or use an existing [Amazon S3 bucket](https://docs.aws.amazon.com/AmazonS3/latest/userguide/UsingBucket.html) of your choice. Make sure that this bucket can be read by [AWS CloudFormation](https://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/Welcome.html).
4. Create the Lambda layer file named `py312_opensearch-py_requests_and_requests-aws4auth.zip` using the following procedure and upload it to the same Amazon S3 bucket as in step 3.
   - On Windows 10 or above:
     1. Make sure [Python 3.12](https://www.python.org/downloads/release/python-3120/) and [pip](https://pip.pypa.io/en/stable/installation/) are installed and set in the user's PATH variable.
     2. Download [7-zip](https://www.7-zip.org/) and install it in `C:/Program Files/7-Zip/`.
     3. Open the Windows command prompt.
     4. Create a new directory and `cd` into it.
     5. Run the [lambda_layer_file_create.bat](https://github.com/aws-samples/advanced-rag-router-with-amazon-bedrock/blob/main/assets/dependencies/lambda_layer_file_create.bat) from inside of that directory.
     6. This will create the Lambda layer file named `py312_opensearch-py_requests_and_requests-aws4auth.zip`.
   - On Linux:
     1. Make sure [Python 3.12](https://www.python.org/downloads/release/python-3120/) and [pip](https://pip.pypa.io/en/stable/installation/) are installed and set in the user's PATH variable.
     2. Open the Linux command prompt.
     3. Create a new directory and `cd` into it.
     4. Run the [lambda_layer_file_create.sh](https://github.com/aws-samples/advanced-rag-router-with-amazon-bedrock/blob/main/assets/dependencies/lambda_layer_file_create.sh) from inside of that directory.
     5. This will create the Lambda layer file named `py312_opensearch-py_requests_and_requests-aws4auth.zip`.
5. Take the provided AWS CloudFormation template [rag-router-cfn.yaml](https://github.com/aws-samples/advanced-rag-router-with-amazon-bedrock/blob/main/assets/rag-router-cfn.yaml) and update the following parameter,
   * *DeploymentArtifactsS3BucketName* - set this to the name of the Amazon S3 bucket from step 3.
6. Create an [AWS CloudFormation stack](https://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/cfn-whatis-concepts.html#cfn-concepts-stacks) with the updated template.
7. Open the Jupyter notebook named *rag-router.ipynb* by navigating to the [Amazon SageMaker notebook instances console](https://docs.aws.amazon.com/sagemaker/latest/dg/howitworks-access-ws.html) and clicking on the *Open Jupyter* link on the instance named *rag-router-instance*.

### Repository structure

This repository contains

* [A Jupyter Notebook](https://github.com/aws-samples/advanced-rag-router-with-amazon-bedrock/blob/main/notebooks/rag-router.ipynb) to get started.

* [A set of helper functions for the notebook](https://github.com/aws-samples/advanced-rag-router-with-amazon-bedrock/blob/main/notebooks/scripts/helper_functions.py)

* [Architecture diagrams](https://github.com/aws-samples/advanced-rag-router-with-amazon-bedrock/blob/main/notebooks/images/) that show the various components used in this session along with their interactions.

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the LICENSE file.

