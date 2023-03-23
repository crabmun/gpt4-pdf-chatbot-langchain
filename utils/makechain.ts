import { OpenAIChat } from 'langchain/llms';
import { LLMChain, ChatVectorDBQAChain, loadQAChain } from 'langchain/chains';
import { PineconeStore } from 'langchain/vectorstores';
import { PromptTemplate } from 'langchain/prompts';
import { CallbackManager } from 'langchain/callbacks';

const CONDENSE_PROMPT =
  PromptTemplate.fromTemplate(`Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:`);

const QA_PROMPT = PromptTemplate.fromTemplate(
  `You are Kate, a friendly and knowledgeable clinical psychologist who 
  specializes in cognitive behavior therapy. As an AI assistant, your job is to 
  provide helpful advice to your clients by answering their questions and 
  responding to their statements about how they are feeling and maintaining a 
  conversational tone. You have been given the following extracted parts of a 
  document related to cognitive behavior therapy. Use the document to help 
  asking questions and give advice. Using your expertise, provide a 
  conversational answer based on the context provided. You should not provide
   any links. If you can't find the answer in the context, just say "Hmm, I'm not 
  sure., can you rephrase?" and wait for the human to rephrase the question or 
  statement. If the question is not related to the context, politely respond that 
  you are tuned to only answer questions that are related to cognitive behavior 
  therapy.

Question: {question}
=========
{context}
=========
Answer in Markdown:`,
);

export const makeChain = (
  vectorstore: PineconeStore,
  onTokenStream?: (token: string) => void,
) => {
  const questionGenerator = new LLMChain({
    llm: new OpenAIChat({ temperature: 0 }),
    prompt: CONDENSE_PROMPT,
  });
  const docChain = loadQAChain(
    new OpenAIChat({
      temperature: 0,
      modelName: 'gpt-4', //change this to older versions (e.g. gpt-3.5-turbo) if you don't have access to gpt-4
      streaming: Boolean(onTokenStream),
      callbackManager: onTokenStream
        ? CallbackManager.fromHandlers({
            async handleLLMNewToken(token) {
              onTokenStream(token);
              console.log(token);
            },
          })
        : undefined, // THIS IS AT THE END OF EACH PARAGRAPH!!
    }),
    { prompt: QA_PROMPT },
  );

  return new ChatVectorDBQAChain({
    vectorstore,
    combineDocumentsChain: docChain,
    questionGeneratorChain: questionGenerator,
    returnSourceDocuments: false,
    k: 2, //number of source documents to return I HAVE CHANGED THIS TO FALSE, if need source, make it "TRUE"
  });
};
