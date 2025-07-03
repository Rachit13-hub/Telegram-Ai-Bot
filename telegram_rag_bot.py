import os
import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Any

# Telegram Bot
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, CallbackQueryHandler

# LangChain imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain.memory import ConversationBufferWindowMemory
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

# Other imports
import requests
import wikipedia
from pathlib import Path

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class TelegramAIBot:
    def __init__(self, telegram_token: str, google_api_key: str):
        self.telegram_token = telegram_token
        self.google_api_key = google_api_key
        
        # Initialize LLM
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=google_api_key,
            temperature=0.7
        )
        
        # Initialize embeddings for RAG
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=google_api_key
        )
        
        # Memory for each user
        self.user_memories: Dict[int, ConversationBufferWindowMemory] = {}
        
        # Vector store for RAG
        self.vector_store = None
        self.retrieval_chain = None
        
        # Store PDF metadata for users
        self.user_pdfs: Dict[int, List[Dict]] = {}
        
        # Initialize tools
        self.tools = self._create_tools()
        
        # Initialize agent
        self.agent_executor = self._create_agent()
        
        # Create documents directory
        Path("documents").mkdir(exist_ok=True)
        Path("documents/pdfs").mkdir(exist_ok=True)
        
        # Load initial knowledge base
        self._load_knowledge_base()
    
    def _get_user_memory(self, user_id: int) -> ConversationBufferWindowMemory:
        """Get or create memory for a specific user"""
        if user_id not in self.user_memories:
            self.user_memories[user_id] = ConversationBufferWindowMemory(
                k=10,  # Remember last 10 exchanges
                return_messages=True
            )
        return self.user_memories[user_id]
    
    def _get_user_pdfs(self, user_id: int) -> List[Dict]:
        """Get list of PDFs uploaded by user"""
        if user_id not in self.user_pdfs:
            self.user_pdfs[user_id] = []
        return self.user_pdfs[user_id]
    
    def _create_tools(self) -> List[Tool]:
        """Create tools for the agent"""
        def search_wikipedia(query: str) -> str:
            """Search Wikipedia for information"""
            try:
                result = wikipedia.summary(query, sentences=3)
                return f"Wikipedia: {result}"
            except Exception as e:
                return f"Wikipedia search failed: {str(e)}"
        
        def calculate(expression: str) -> str:
            """Perform mathematical calculations"""
            try:
                result = eval(expression)
                return f"Result: {result}"
            except Exception as e:
                return f"Calculation error: {str(e)}"
        
        def get_current_time() -> str:
            """Get current date and time"""
            return f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        def search_pdf_content(query: str) -> str:
            """Search through uploaded PDF content"""
            if self.vector_store:
                try:
                    # Use the retrieval chain to search PDF content
                    result = self.retrieval_chain.invoke({"query": query})
                    return f"Search Result: {result['result']}"
                except Exception as e:
                    return f"Document search failed: {str(e)}"
            return "No document content available to search"
        
        return [
            Tool(
                name="wikipedia_search",
                description="Search Wikipedia for information about any topic",
                func=search_wikipedia
            ),
            Tool(
                name="calculator",
                description="Perform mathematical calculations",
                func=calculate
            ),
            Tool(
                name="current_time",
                description="Get current date and time",
                func=get_current_time
            ),
            Tool(
                name="document_search",
                description="Search through uploaded documents and knowledge base for specific information",
                func=search_pdf_content
            )
        ]
    
    def _create_agent(self) -> AgentExecutor:
        """Create a ReAct agent with tools"""
        prompt = PromptTemplate.from_template("""
You are a helpful AI assistant with access to various tools. Answer the following questions as best you can.

You have access to the following tools:
{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Question: {input}
Thought: {agent_scratchpad}
""")
        
        agent = create_react_agent(self.llm, self.tools, prompt)
        return AgentExecutor(agent=agent, tools=self.tools, verbose=True, max_iterations=3)
    
    def _load_knowledge_base(self):
        """Load and process documents for RAG"""
        documents = []
        doc_dir = Path("documents")
        
        # Create a sample knowledge base if no documents exist
        if not any(doc_dir.iterdir()):
            sample_content = """
            This is the AI Assistant Knowledge Base.
            
            About this bot:
            - I'm an AI assistant built with LangChain and Google's Gemini AI
            - I can help with various tasks including answering questions, calculations,and more
            - I remember our conversation history during this session
            - I can search through uploaded documents and PDFs to provide better answers
            - I can read and analyze PDF documents you upload
            
            Available commands:
            /start - Start the bot
            /help - Show help information
            /clear - Clear conversation memory
            /upload - Upload documents for RAG
            /stats - Show bot statistics
            /mypdfs - Show your uploaded PDFs
            """
            
            with open(doc_dir / "knowledge_base.txt", "w") as f:
                f.write(sample_content)
        
        # Load text documents
        for file_path in doc_dir.glob("*.txt"):
            try:
                loader = TextLoader(str(file_path))
                text_docs = loader.load()
                # Add metadata to identify text file source
                for doc in text_docs:
                    doc.metadata['source_type'] = 'text'
                    doc.metadata['file_name'] = file_path.name
                documents.extend(text_docs)
                logger.info(f"Loaded text file: {file_path.name}")
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
        
        # Load PDF documents
        pdf_dir = doc_dir / "pdfs"
        for file_path in pdf_dir.glob("*.pdf"):
            try:
                loader = PyPDFLoader(str(file_path))
                pdf_docs = loader.load()
                # Add metadata to identify PDF source
                for doc in pdf_docs:
                    doc.metadata['source_type'] = 'pdf'
                    doc.metadata['file_name'] = file_path.name
                documents.extend(pdf_docs)
                logger.info(f"Loaded PDF: {file_path.name} with {len(pdf_docs)} pages")
            except Exception as e:
                logger.error(f"Error loading PDF {file_path}: {e}")
        
        if documents:
            # Split documents
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            splits = text_splitter.split_documents(documents)
            
            # Create vector store
            self.vector_store = FAISS.from_documents(splits, self.embeddings)
            
            # Create retrieval chain
            retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})
            
            self.retrieval_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True
            )
            
            logger.info(f"Loaded {len(splits)} document chunks into vector store")
    
    def _get_source_attribution(self, source_documents) -> str:
        """Get proper source attribution from retrieved documents"""
        if not source_documents:
            return ""
        
        text_sources = []
        pdf_sources = []
        
        for doc in source_documents:
            source_type = doc.metadata.get('source_type', 'unknown')
            file_name = doc.metadata.get('file_name', 'unknown')
            
            if source_type == 'pdf':
                pdf_sources.append(file_name)
            elif source_type == 'text':
                text_sources.append(file_name)
        
        # Remove duplicates
        text_sources = list(set(text_sources))
        pdf_sources = list(set(pdf_sources))
        
        attribution = ""
        if text_sources:
            attribution += f"Knowledge Base: {', '.join(text_sources)}"
        if pdf_sources:
            if attribution:
                attribution += " | "
            attribution += f"PDFs: {', '.join(pdf_sources)}"
        
        return attribution
    
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command"""
        keyboard = [
            [InlineKeyboardButton("🤖 Chat with AI", callback_data="chat")],
            [InlineKeyboardButton("📚 Knowledge Search", callback_data="search")],
            [InlineKeyboardButton("📄 Upload PDF", callback_data="upload_pdf")],
            [InlineKeyboardButton("🛠️ Tools", callback_data="tools")],
            [InlineKeyboardButton("ℹ️ Help", callback_data="help")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        welcome_message = """
🤖 Welcome to AI Assistant Bot!

I'm powered by Google's Gemini AI and built with LangChain. Here's what I can do:

✨ **Features:**
• Intelligent conversations with memory
• Document search (RAG)
• **PDF Reading & Analysis** 📄
• Wikipedia search
• Mathematical calculations
• Time queries

🚀 **Get Started:**
Just send me a message, upload a PDF, or use the buttons below!
        """
        
        await update.message.reply_text(welcome_message, reply_markup=reply_markup)
    
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command"""
        help_text = """
🤖 **AI Assistant Bot Help**

**Available Commands:**
/start - Start the bot
/help - Show this help message
/clear - Clear conversation memory
/stats - Show bot statistics
/mypdfs - Show your uploaded PDFs

**Features:**
• **Chat:** Just send any message for AI conversation
• **Memory:** I remember our conversation during this session
• **PDF Reading:** Upload PDFs and ask questions about them
• **RAG:** I can search through uploaded documents
• **Tools:** I can use Wikipedia, calculator,  and time tools
• **Agent:** I can reason and use multiple tools to answer complex questions

**PDF Examples:**
• Upload a PDF, then ask: "What is this document about?"
• "Summarize the main points from the PDF"
• "Find information about [topic] in the uploaded document"
• "What are the key conclusions in this paper?"

**Other Examples:**
• "Calculate 25 * 67 + 123"
• "Search for information about artificial intelligence"
• "What time is it?"

Just start chatting or upload a PDF! 💬📄
        """
        await update.message.reply_text(help_text)
    
    async def clear_memory_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /clear command"""
        user_id = update.effective_user.id
        if user_id in self.user_memories:
            self.user_memories[user_id].clear()
            await update.message.reply_text("🧹 Conversation memory cleared!")
        else:
            await update.message.reply_text("No memory found to clear.")
    
    async def stats_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /stats command"""
        user_id = update.effective_user.id
        memory = self._get_user_memory(user_id)
        user_pdfs = self._get_user_pdfs(user_id)
        
        stats = f"""
📊 **Bot Statistics**

**Your Session:**
• Messages in memory: {len(memory.chat_memory.messages)}
• PDFs uploaded: {len(user_pdfs)}
• Vector store loaded: {'✅' if self.vector_store else '❌'}
• Available tools: {len(self.tools)}

**System Info:**
• Model: Google Gemini Pro
• Framework: LangChain
• Memory window: 10 exchanges
• PDF Support: ✅ Enabled
        """
        await update.message.reply_text(stats)
    
    async def mypdfs_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /mypdfs command"""
        user_id = update.effective_user.id
        user_pdfs = self._get_user_pdfs(user_id)
        
        if not user_pdfs:
            await update.message.reply_text("📄 No PDFs uploaded yet. Send me a PDF file to get started!")
            return
        
        pdfs_text = "📄 **Your Uploaded PDFs:**\n\n"
        for i, pdf in enumerate(user_pdfs, 1):
            pdfs_text += f"{i}. **{pdf['name']}**\n"
            pdfs_text += f"   • Size: {pdf['size']} bytes\n"
            pdfs_text += f"   • Uploaded: {pdf['uploaded_at']}\n"
            pdfs_text += f"   • Pages: {pdf.get('pages', 'Unknown')}\n\n"
        
        pdfs_text += "💡 You can ask me questions about any of these PDFs!"
        await update.message.reply_text(pdfs_text)
    
    async def handle_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle inline keyboard callbacks"""
        query = update.callback_query
        await query.answer()
        
        if query.data == "chat":
            await query.edit_message_text("💬 Great! Just send me any message and I'll respond using AI!")
        elif query.data == "search":
            await query.edit_message_text("🔍 Send me a question and I'll search through my knowledge base and PDFs!")
        elif query.data == "upload_pdf":
            await query.edit_message_text("📄 Send me a PDF file and I'll process it so you can ask questions about it!")
        elif query.data == "tools":
            tools_text = "🛠️ **Available Tools:**\n\n"
            for tool in self.tools:
                tools_text += f"• **{tool.name}**: {tool.description}\n"
            await query.edit_message_text(tools_text)
        elif query.data == "help":
            await self.help_command(update, context)
    
    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle regular messages"""
        user_id = update.effective_user.id
        user_message = update.message.text
        
        # Show typing indicator
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
        
        try:
            # Get user memory
            memory = self._get_user_memory(user_id)
            user_pdfs = self._get_user_pdfs(user_id)
            
            # Prepare context with memory
            chat_history = ""
            if memory.chat_memory.messages:
                for msg in memory.chat_memory.messages[-6:]:  # Last 3 exchanges
                    if isinstance(msg, HumanMessage):
                        chat_history += f"Human: {msg.content}\n"
                    elif isinstance(msg, AIMessage):
                        chat_history += f"AI: {msg.content}\n"
            
            # First, try to use RAG if available
            rag_context = ""
            source_info = ""
            if self.retrieval_chain:
                try:
                    rag_result = self.retrieval_chain.invoke({"query": user_message})
                    rag_context = f"\nRelevant context from documents:\n{rag_result['result']}\n"
                    
                    # Get proper source attribution
                    if 'source_documents' in rag_result and rag_result['source_documents']:
                        source_info = self._get_source_attribution(rag_result['source_documents'])
                    
                except Exception as e:
                    logger.error(f"RAG search failed: {e}")
            
            # Check if user is asking about PDFs specifically
            pdf_keywords = ["pdf", "document", "paper", "file", "upload"]
            is_pdf_query = any(keyword in user_message.lower() for keyword in pdf_keywords)
            
            # Add PDF context if user has uploaded PDFs
            pdf_context = ""
            if user_pdfs and is_pdf_query:
                pdf_names = [pdf['name'] for pdf in user_pdfs]
                pdf_context = f"\nAvailable PDFs: {', '.join(pdf_names)}\n"
            
            # Determine if we should use the agent or simple chat
            agent_keywords = ["search", "calculate", "time", "wikipedia", "what time", "find in pdf", "search pdf","search from knowledge base"]
            use_agent = any(keyword in user_message.lower() for keyword in agent_keywords)
            
            if use_agent:
                # Use agent for tool-based queries
                response = self.agent_executor.invoke({"input": user_message})
                ai_response = response["output"]
            else:
                # Decide whether to use RAG based on relevance or keywords
                use_rag = any(keyword in user_message.lower() for keyword in ["find", "search", "info", "document", "based on"])
                rag_context = ""
                source_info = ""

                if use_rag and self.retrieval_chain:
                    try:
                        rag_result = self.retrieval_chain.invoke({"query": user_message})
                        rag_context = f"\nRelevant context from documents:\n{rag_result['result']}\n"

                        if 'source_documents' in rag_result and rag_result['source_documents']:
                            source_info = self._get_source_attribution(rag_result['source_documents'])
                    except Exception as e:
                        logger.error(f"RAG search failed: {e}")

                system_prompt = f"""
            You are a helpful AI assistant.

            {chat_history}
            {rag_context}
            {pdf_context}

            Answer the user's message naturally and conversationally. If you reference documents, mention the source.
            """

                messages = [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_message)
                ]

                response = self.llm.invoke(messages)
                ai_response = response.content

            
            # Add source attribution if available
            if source_info and not use_agent:
                ai_response += f"\n\n(Source: {source_info})"
            
            # Store in memory
            memory.chat_memory.add_user_message(user_message)
            memory.chat_memory.add_ai_message(ai_response)
            
            # Send response
            await update.message.reply_text(ai_response)
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            await update.message.reply_text(
                "Sorry, I encountered an error processing your message. Please try again."
            )
    
    async def handle_document(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle document uploads for RAG"""
        user_id = update.effective_user.id
        document = update.message.document
        
        # Show upload progress
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="upload_document")
        
        try:
            if document.mime_type == 'application/pdf':
                # Handle PDF upload
                await self._handle_pdf_upload(update, context, document)
            elif document.mime_type == 'text/plain':
                # Handle text file upload
                file = await context.bot.get_file(document.file_id)
                file_path = f"documents/{document.file_name}"
                await file.download_to_drive(file_path)
                
                # Reload knowledge base
                self._load_knowledge_base()
                
                await update.message.reply_text(
                    f"✅ Text document '{document.file_name}' uploaded and processed!"
                )
            else:
                await update.message.reply_text(
                    "📄 Please upload PDF files (.pdf) or text files (.txt). "
                    "I can read and analyze PDF documents!"
                )
        except Exception as e:
            logger.error(f"Error handling document upload: {e}")
            await update.message.reply_text(
                "❌ Sorry, there was an error processing your document. Please try again."
            )
    
    async def _handle_pdf_upload(self, update: Update, context: ContextTypes.DEFAULT_TYPE, document):
        """Handle PDF document upload"""
        user_id = update.effective_user.id
        
        try:
            # Download the PDF file
            file = await context.bot.get_file(document.file_id)
            file_path = f"documents/pdfs/{document.file_name}"
            await file.download_to_drive(file_path)
            
            # Send processing message
            processing_msg = await update.message.reply_text("📄 Processing PDF... This may take a moment.")
            
            # Load and process the PDF
            loader = PyPDFLoader(file_path)
            pages = loader.load()
            
            # Update processing message
            await processing_msg.edit_text("🔄 Analyzing PDF content and creating embeddings...")
            
            # Add to user's PDF list
            user_pdfs = self._get_user_pdfs(user_id)
            pdf_info = {
                'name': document.file_name,
                'size': document.file_size,
                'uploaded_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'pages': len(pages),
                'file_path': file_path
            }
            user_pdfs.append(pdf_info)
            
            # Reload knowledge base to include new PDF
            self._load_knowledge_base()
            
            # Create summary of the PDF
            if pages:
                # Get first few pages for summary
                first_pages_content = "\n".join([page.page_content for page in pages[:3]])
                
                summary_prompt = f"""
Please provide a brief summary of this PDF document based on the first few pages:

{first_pages_content[:2000]}...

Summary should include:
1. Main topic/subject
2. Type of document (research paper, report, manual, etc.)
3. Key themes or sections
"""
                
                try:
                    summary_response = self.llm.invoke([AIMessage(content=summary_prompt)])
                    summary = summary_response.content
                except:
                    summary = "Summary generation failed, but PDF is ready for questions."
            else:
                summary = "PDF processed but appears to be empty or unreadable."
            
            # Send success message with summary
            success_message = f"""
✅ **PDF Successfully Processed!**

📄 **File:** {document.file_name}
📊 **Pages:** {len(pages)}
📏 **Size:** {document.file_size} bytes

**Quick Summary:**
{summary}

💡 **You can now ask me questions about the uploaded PDF!**

Examples:
• "What is this document about?"
• "Summarize the main points"
• "Find information about [specific topic]"
• "What are the key conclusions?"
"""
            
            await processing_msg.edit_text(success_message)
            
        except Exception as e:
            logger.error(f"Error processing PDF: {e}")
            await update.message.reply_text(
                f"❌ Error processing PDF '{document.file_name}': {str(e)}\n"
                "Please make sure the PDF is readable and try again."
            )
    
    def run(self):
        """Run the bot"""
        application = Application.builder().token(self.telegram_token).build()
        
        # Command handlers
        application.add_handler(CommandHandler("start", self.start_command))
        application.add_handler(CommandHandler("help", self.help_command))
        application.add_handler(CommandHandler("clear", self.clear_memory_command))
        application.add_handler(CommandHandler("stats", self.stats_command))
        application.add_handler(CommandHandler("mypdfs", self.mypdfs_command))
        
        # Callback handler
        application.add_handler(CallbackQueryHandler(self.handle_callback))
        
        # Message handlers
        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))
        application.add_handler(MessageHandler(filters.Document.ALL, self.handle_document))
        
        # Start the bot
        logger.info("Starting Telegram AI Bot with PDF support...")
        application.run_polling()

def main():
    # Configuration - REPLACE WITH YOUR ACTUAL TOKENS
    TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN', 'Your_telegram_bot_token')
    GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY', "Your_api_key")
    
    # Validate tokens
    if TELEGRAM_TOKEN == "YOUR_TELEGRAM_BOT_TOKEN" or GOOGLE_API_KEY == "YOUR_GOOGLE_API_KEY":
        print("❌ Please set your TELEGRAM_TOKEN and GOOGLE_API_KEY in environment variables or update the code")
        return
    
    # Create and run bot
    bot = TelegramAIBot(TELEGRAM_TOKEN, GOOGLE_API_KEY)
    bot.run()

if __name__ == "__main__":
    main()
