#!/usr/bin/env python3
"""
Main orchestrator script for the vast.ai automation system.

This script provides a clean, modular interface to the vast.ai automation system.
It uses the new layered architecture with reusable primitive functions.
"""

import argparse
import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from density_engine.vast.orchestration.workflow import get_workflow_manager
from density_engine.vast.utils.logging import setup_logging, get_logger
from density_engine.vast.utils.config import load_config, validate_config

logger = get_logger(__name__)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Vast.ai Automation Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run the orchestrator for 10 iterations (cache cleared by default)
  python orchestrator.py --run --max-iterations 10
  
  # Show current status
  python orchestrator.py --status
  
  # Start orchestrator and run indefinitely (cache cleared by default)
  python orchestrator.py --run
  
  # Keep cached state and run
  python orchestrator.py --run --keep-cache
        """
    )
    
    # Main commands
    parser.add_argument(
        '--run',
        action='store_true',
        help='Run the orchestrator'
    )
    
    parser.add_argument(
        '--status',
        action='store_true',
        help='Show current status'
    )
    
    parser.add_argument(
        '--max-iterations',
        type=int,
        default=None,
        help='Maximum number of iterations to run (default: unlimited)'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Log level (default: INFO)'
    )
    
    parser.add_argument(
        '--config-file',
        type=Path,
        default=None,
        help='Configuration file path'
    )
    
    parser.add_argument(
        '--keep-cache',
        action='store_true',
        help='Keep cached state (default: clear cache on startup)'
    )
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.log_level)
    
    # Clear cache by default (unless --keep-cache is specified)
    if not args.keep_cache:
        logger.info("Clearing cached state (default behavior)...")
        try:
            from density_engine.vast.core.state import get_state_manager
            state_manager = get_state_manager()
            state_manager.clear_state()
            logger.info("✅ Cached state cleared successfully")
        except Exception as e:
            logger.error(f"Failed to clear cached state: {e}")
            return 1
    else:
        logger.info("Keeping cached state (--keep-cache specified)")
    
    # Load configuration
    try:
        config = load_config(args.config_file)
        errors = validate_config(config)
        if errors:
            logger.error("Configuration validation failed:")
            for error in errors:
                logger.error(f"  - {error}")
            return 1
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return 1
    
    # Get workflow manager
    workflow_manager = get_workflow_manager()
    
    try:
        if args.status:
            # Show status
            workflow_manager.print_workflow_status()
            return 0
        
        elif args.run:
            # Run the orchestrator
            logger.info("Starting vast.ai automation orchestrator")
            logger.info(f"Max iterations: {args.max_iterations or 'unlimited'}")
            
            success = workflow_manager.run_workflow(max_iterations=args.max_iterations)
            
            if success:
                logger.info("✅ Orchestrator completed successfully")
                return 0
            else:
                logger.error("❌ Orchestrator failed")
                return 1
        
        else:
            # No command specified
            parser.print_help()
            return 1
    
    except KeyboardInterrupt:
        logger.info("Orchestrator interrupted by user")
        workflow_manager.stop_workflow()
        return 0
    
    except Exception as e:
        logger.error(f"Orchestrator failed with error: {e}")
        workflow_manager.stop_workflow()
        return 1


if __name__ == "__main__":
    sys.exit(main())
