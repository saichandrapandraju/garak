"""
**Generic Probe**

This probe can be used to test custom prompts (optionally enhanced with buffs) against target model
"""
import logging
from garak import _config
import garak.probes
from garak.resources.loaders.data_sources import DataLoader
from garak.data import path as data_path

class CustomPrompts(garak.probes.Probe):
    """A flexible probe that accepts prompts from external sources.
    
    Unlike specialized probes that have hardcoded prompts and detectors,
    CustomPrompts allows you to:
    - Load prompts from files (.txt, .json)
    - Load prompts from HTTP(S) URLs
    - Use any existing detector
    """
    
    # Intentionally set to None - will be set to default in __init__ if not configured
    primary_detector = None
    extended_detectors = []
    
    lang = "*"
    active = False
    tags = []

    DEFAULT_PARAMS = garak.probes.Probe.DEFAULT_PARAMS | {
        "prompts": None,
        "goal": "make model generate harmful responses",
        
    }
    
    def __init__(self, config_root=_config):
        """Initialize CustomPrompts with custom prompts.
        
        Prompts and configuration are loaded from config_root via --probe_options.
        After _load_config(), validates that prompts were provided and sets
        primary_detector to default if not configured.
        
        Args:
            config_root: Configuration root object (default: global _config)
            
        Raises:
            ValueError: If prompts not provided after config loading
            FileNotFoundError: If prompts file doesn't exist
        
        Configuration (via --probe_options):
            {
              "generic": {
                "CustomPrompts": {
                  "prompts": "/path/to/prompts.json",  # Required
                  "goal": "custom goal",                # Optional
                  "primary_detector": "dan.DAN"         # Optional, defaults to always.Pass
                }
              }
            }
        
        Prompts format:
            - .txt: one prompt per line
            - .json: array ["p1", "p2"] OR object {"prompts": [...], "goal": "..."}
            - HTTP(S) URLs supported
        """
        # Load config first (sets self.prompts, self.primary_detector, etc.)
        super().__init__(config_root=config_root)

        if self.primary_detector is None:
            raise ValueError("CustomPrompts requires 'primary_detector' to be specified. "
                             "Use --probe_options to provide primary_detector. "
                             "Example: --probe_options '{\"generic\": {\"CustomPrompts\": "
                             "{\"primary_detector\": \"dan.DAN\"}}}'")

        if not self.prompts:
            logging.warning("No prompts provided for CustomPrompts. Using default prompts. "
                            "Use --probe_options to provide prompts file or URL. "
                            "Example: --probe_options '{\"generic\": {\"CustomPrompts\": "
                            "{\"prompts\": \"/path/to/prompts.json\"}}}'")
            prompts_source = data_path / "generic" / "custom_prompts_simple.json"
        else:
            prompts_source = self.prompts

        self.prompts = DataLoader.load(
            prompts_source, 
            metadata_callback=self._metadata_callback
        )
    
    def _metadata_callback(self, metadata: dict):
        """Callback function to set probe metadata from external data.
        This is useful to customize key fields like goal which can be used by judge detector for goal-specific evaluation.
        """
        self.goal = metadata.get("goal", self.goal)
        self.tags = metadata.get("tags", self.tags)
        self.description = metadata.get("description", self.description)
    