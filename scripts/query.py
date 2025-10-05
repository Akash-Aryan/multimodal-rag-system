#!/usr/bin/env python3
"""
Query script for the multimodal RAG system.
Usage: python scripts/query.py --query "What are the main topics?" --config config/config.yaml
"""

import asyncio
import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.multimodal_rag import MultimodalRAG, RAGConfig
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown


async def main():
    parser = argparse.ArgumentParser(description="Query the multimodal RAG system")
    parser.add_argument("--query", "-q", required=True, help="Query text")
    parser.add_argument("--config", "-c", default="config/config.yaml", help="Configuration file path")
    parser.add_argument("--query-type", "-t", default="text", choices=["text", "image", "audio"], 
                       help="Type of query")
    parser.add_argument("--top-k", "-k", type=int, default=5, help="Number of results to retrieve")
    parser.add_argument("--no-sources", action="store_true", help="Don't include sources in output")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive query mode")
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        config = RAGConfig.from_yaml(args.config)
        
        # Initialize RAG system
        rag_system = MultimodalRAG(config)
        
        # Initialize rich console for better output
        console = Console()
        
        if args.interactive:
            # Interactive mode
            console.print(Panel("🤖 Multimodal RAG Interactive Query Mode", style="bold blue"))
            console.print("Type 'exit' or 'quit' to stop, 'stats' for system statistics\\n")
            
            while True:
                try:
                    query = input("❓ Enter your query: ").strip()
                    
                    if query.lower() in ['exit', 'quit']:
                        console.print("👋 Goodbye!", style="bold green")
                        break
                    
                    if query.lower() == 'stats':
                        stats = await rag_system.get_system_stats()
                        console.print(Panel(str(stats), title="📊 System Statistics"))
                        continue
                    
                    if not query:
                        continue
                    
                    # Process query
                    console.print(f"🔍 Processing query: {query}")
                    result = await rag_system.query(
                        query=query,
                        query_type=args.query_type,
                        top_k=args.top_k,
                        include_sources=not args.no_sources
                    )
                    
                    # Display response
                    console.print(Panel(
                        Markdown(result['response']),
                        title="🤖 AI Response",
                        border_style="green"
                    ))
                    
                    # Display sources if included
                    if not args.no_sources and result.get('sources'):
                        console.print("\\n📚 **Sources:**")
                        for i, source in enumerate(result['sources'][:3], 1):  # Show top 3 sources
                            doc = source.get('document', {})
                            metadata = doc.get('metadata', {})
                            source_file = metadata.get('source_file', 'Unknown')
                            score = source.get('score', 0)
                            
                            console.print(f"  {i}. **{Path(source_file).name}** (similarity: {score:.3f})")
                            if doc.get('text'):
                                preview = doc['text'][:100] + "..." if len(doc['text']) > 100 else doc['text']
                                console.print(f"     _{preview}_")
                    
                    console.print("\\n" + "="*50 + "\\n")
                
                except KeyboardInterrupt:
                    console.print("\\n👋 Goodbye!", style="bold green")
                    break
                except Exception as e:
                    console.print(f"❌ Error: {e}", style="bold red")
        
        else:
            # Single query mode
            logger.info(f"Processing query: {args.query}")
            
            result = await rag_system.query(
                query=args.query,
                query_type=args.query_type,
                top_k=args.top_k,
                include_sources=not args.no_sources
            )
            
            # Display results using rich formatting
            console.print(Panel(
                Markdown(result['response']),
                title="🤖 AI Response",
                border_style="green"
            ))
            
            # Display metadata
            console.print(f"\\n📊 **Query Stats:**")
            console.print(f"  - Query type: {result.get('query_type', 'unknown')}")
            console.print(f"  - Retrieved documents: {result.get('retrieved_count', 0)}")
            console.print(f"  - Confidence scores: {result.get('confidence_scores', [])[:3]}")
            
            # Display sources if included
            if not args.no_sources and result.get('sources'):
                console.print("\\n📚 **Sources:**")
                for i, source in enumerate(result['sources'], 1):
                    doc = source.get('document', {})
                    metadata = doc.get('metadata', {})
                    source_file = metadata.get('source_file', 'Unknown')
                    score = source.get('score', 0)
                    
                    console.print(f"  {i}. **{Path(source_file).name}** (similarity: {score:.3f})")
                    if doc.get('text'):
                        preview = doc['text'][:150] + "..." if len(doc['text']) > 150 else doc['text']
                        console.print(f"     _{preview}_\\n")
        
    except Exception as e:
        logger.error(f"Error during query: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
