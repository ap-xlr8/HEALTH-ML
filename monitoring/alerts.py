"""Alert Dispatcher Module for Health OS ML Operations."""

from __future__ import annotations

import os
import logging
import requests
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class AlertDispatcher:
    """Dispatches operational alerts to Slack, PagerDuty, or Webhook endpoints."""

    @staticmethod
    def send_drift_alert(
        drift_results: Dict[str, Any],
        slack_webhook_url: Optional[str] = None,
        pagerduty_key: Optional[str] = None,
    ) -> bool:
        """Send alert if data or model drift has exceeded threshold."""
        slack_url = slack_webhook_url or os.environ.get("SLACK_WEBHOOK_URL")
        pd_key = pagerduty_key or os.environ.get("PAGERDUTY_ROUTING_KEY")

        drifted = drift_results.get("drifted_features", [])
        message = (
            f":warning: *[Health OS ML Alert]* Data Drift Detected!\n"
            f"*Drifted Features:* {', '.join(drifted)}\n"
            f"*Drift Share:* {drift_results.get('drift_share', 0) * 100:.1f}%\n"
            f"*Action Required:* Review production data distribution & trigger re-training."
        )

        sent = False
        if slack_url:
            try:
                res = requests.post(slack_url, json={"text": message}, timeout=3.0)
                if res.status_code == 200:
                    sent = True
                    logger.info("Drift alert sent to Slack successfully.")
            except Exception as e:
                logger.warning("Failed to send Slack alert: %s", e)

        if pd_key and drift_results.get("requires_retraining"):
            try:
                pd_payload = {
                    "routing_key": pd_key,
                    "event_action": "trigger",
                    "payload": {
                        "summary": f"Health OS ML Drift Alert: {len(drifted)} features drifted",
                        "severity": "warning",
                        "source": "health-ml-drift-checker",
                    },
                }
                res = requests.post(
                    "https://events.pagerduty.com/v2/enqueue",
                    json=pd_payload,
                    timeout=3.0,
                )
                if res.status_code in (200, 202):
                    sent = True
                    logger.info("Drift alert sent to PagerDuty successfully.")
            except Exception as e:
                logger.warning("Failed to send PagerDuty alert: %s", e)

        if not sent:
            logger.info("Local Alert Triggered:\n%s", message)

        return sent
