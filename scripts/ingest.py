#!/usr/bin/env python3
"""
Ingestion script for the multimodal RAG system.
Usage: python scripts/ingest.py --directory data/raw --config config/config.yaml
"""

import asyncio
import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.multimodal_rag import MultimodalRAG, RAGConfig
from loguru import logger


async def main():
    parser = argparse.ArgumentParser(description="Ingest documents into the multimodal RAG system")
    parser.add_argument("--directory", "-d", required=True, help="Directory containing documents to ingest")
    parser.add_argument("--config", "-c", default="config/config.yaml", help="Configuration file path")
    parser.add_argument("--file", "-f", help="Single file to ingest")
    parser.add_argument("--save-index", action="store_true", help="Save the index after ingestion")
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        config = RAGConfig.from_yaml(args.config)
        
        # Initialize RAG system
        rag_system = MultimodalRAG(config)
        
        if args.file:
            # Ingest single file
            logger.info(f"Ingesting single file: {args.file}")
            result = await rag_system.ingest_file(args.file)
            logger.info(f"Ingestion result: {result}")
        
        else:
            # Ingest directory
            logger.info(f"Ingesting directory: {args.directory}")
            results = await rag_system.ingest_directory(args.directory)
            
            # Summary statistics
            total_files = len(results)
            successful = sum(1 for r in results if r.get('status') == 'success')
            failed = total_files - successful
            total_chunks = sum(r.get('content_chunks', 0) for r in results if r.get('status') == 'success')
            
            logger.info(f"Ingestion complete!")
            logger.info(f"  Total files processed: {total_files}")
            logger.info(f"  Successful: {successful}")
            logger.info(f"  Failed: {failed}")
            logger.info(f"  Total chunks created: {total_chunks}")
            
            # Show failed files
            if failed > 0:
                logger.warning("Failed files:")
                for result in results:
                    if result.get('status') == 'error':
                        logger.warning(f"  - {result.get('file_path', 'Unknown')}: {result.get('error', 'Unknown error')}")
        
        # Save index if requested
        if args.save_index:
            logger.info("Saving index...")
            await rag_system.save_index()
            logger.info("Index saved successfully!")
        
        # Show system stats
        stats = await rag_system.get_system_stats()
        logger.info(f"System stats: {stats}")
        
    except Exception as e:
        logger.error(f"Error during ingestion: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
