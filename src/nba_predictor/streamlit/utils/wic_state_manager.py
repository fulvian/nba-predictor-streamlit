"""
WIC State Manager
Manages the session state for the Workflow Intelligent Control Dashboard.
"""

import streamlit as st
from typing import Any, Dict, Optional


class WICState:
    """
    Manages the state for the WIC Dashboard workflow.
    """

    # Keys for session state
    KEY_STEP = "wic_step"
    KEY_SELECTED_GAME_ID = "wic_selected_game_id"
    KEY_SELECTED_GAME_DATA = "wic_selected_game_data"
    KEY_PREDICTION = "wic_prediction"
    KEY_MANUAL_ODDS = "wic_manual_odds"
    KEY_RECOMMENDED_BET = "wic_recommended_bet"

    @staticmethod
    def initialize():
        """Initialize the WIC state if not present."""
        if WICState.KEY_STEP not in st.session_state:
            st.session_state[WICState.KEY_STEP] = 1  # Start at Step 1 (Scheduler)

    @staticmethod
    def get_current_step() -> int:
        return st.session_state.get(WICState.KEY_STEP, 1)

    @staticmethod
    def set_step(step: int):
        st.session_state[WICState.KEY_STEP] = step

    @staticmethod
    def next_step():
        current = WICState.get_current_step()
        if current < 5:
            st.session_state[WICState.KEY_STEP] = current + 1

    @staticmethod
    def prev_step():
        current = WICState.get_current_step()
        if current > 1:
            st.session_state[WICState.KEY_STEP] = current - 1

    @staticmethod
    def reset():
        """Reset the workflow to the beginning."""
        st.session_state[WICState.KEY_STEP] = 1
        st.session_state.pop(WICState.KEY_SELECTED_GAME_ID, None)
        st.session_state.pop(WICState.KEY_SELECTED_GAME_DATA, None)
        st.session_state.pop(WICState.KEY_PREDICTION, None)
        st.session_state.pop(WICState.KEY_MANUAL_ODDS, None)
        st.session_state.pop(WICState.KEY_RECOMMENDED_BET, None)

    @staticmethod
    def set_selected_game(game_id: str, game_data: Dict[str, Any]):
        st.session_state[WICState.KEY_SELECTED_GAME_ID] = game_id
        st.session_state[WICState.KEY_SELECTED_GAME_DATA] = game_data

    @staticmethod
    def get_selected_game() -> Optional[Dict[str, Any]]:
        return st.session_state.get(WICState.KEY_SELECTED_GAME_DATA)

    @staticmethod
    def set_prediction(prediction: Dict[str, Any]):
        st.session_state[WICState.KEY_PREDICTION] = prediction

    @staticmethod
    def get_prediction() -> Optional[Dict[str, Any]]:
        return st.session_state.get(WICState.KEY_PREDICTION)
