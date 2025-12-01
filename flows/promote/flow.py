"""
Promote anomaly detector to champion status.

This flow orchestrates:
1. Load model to promote
2. Assign champion status
3. Publish promotion event

Triggering (OPTIONAL - human-in-the-loop):
- Default: Listens for 'model_approved' event via @project_trigger
- A human publishes this event after reviewing the evaluated model
- To make fully automated: replace @project_trigger with @trigger_on_finish
- To make manual-only: remove the @project_trigger decorator entirely

Model Lifecycle:
- candidate: Newly trained, not yet evaluated
- evaluated: Passed point-in-time quality gates
- challenger: Running in parallel with champion for trial period
- champion: Primary model, blessed for serving
- retired: Previous champion, replaced by newer version

Promotion Paths:
1. Direct: evaluated -> champion (skip challenger phase)
   Use when: offline evaluation is sufficient, low-risk domain

2. With trial: evaluated -> challenger -> champion
   Use when: need online/batch comparison over time
"""

from metaflow import step, Parameter
from obproject import ProjectFlow
from obproject.project_events import project_trigger

# Import src at module level so Metaflow detects METAFLOW_PACKAGE_POLICY
# and includes it in the code package for remote execution
import src


# OPTIONAL: Remove this decorator for manual-only promotion
# Or replace with @trigger_on_finish(flow='EvaluateDetectorFlow') for full automation
@project_trigger(event="model_approved")
class PromoteDetectorFlow(ProjectFlow):
    """
    Promote anomaly detector to champion status.

    Human-in-the-loop: waits for 'model_approved' event.

    Usage:
        # Manual run (always works)
        python flow.py run
        python flow.py run --version v5

        # Deploy to Argo (waits for event)
        python flow.py --with retry argo-workflows create

        # Trigger via event (after human approval)
        python -c "from obproject.project_events import ProjectEvent; \\
            ProjectEvent('model_approved', 'crypto_anomaly', 'main').publish()"
    """

    version = Parameter(
        "version",
        default="latest",
        help="Model version to promote (e.g., 'latest', 'v5')"
    )

    @step
    def start(self):
        """Load model to promote."""
        from src import registry

        print(f"Project: {self.prj.project}, Branch: {self.prj.branch}")

        # Load model to promote
        try:
            self.model_ref = registry.load_model(
                self.prj.asset, "anomaly_detector", instance=self.version
            )
        except Exception as e:
            print(f"\n[ERROR] No anomaly_detector found")
            raise RuntimeError(f"Model not found: {e}")

        print(f"\nModel to promote:")
        print(f"  Version: v{self.model_ref.version}")
        print(f"  Current status: {self.model_ref.status.value}")
        print(f"  Algorithm: {self.model_ref.annotations.get('algorithm')}")

        anomaly_rate = float(self.model_ref.annotations.get("anomaly_rate", 0))
        print(f"  Anomaly rate: {anomaly_rate:.1%}")

        self.next(self.assign_champion)

    @step
    def assign_champion(self):
        """Assign champion status to model."""
        from src import registry
        from metaflow import current

        print(f"\nPromoting to champion...")

        registry.promote_to_champion(
            self.prj.asset,
            "anomaly_detector",
            source_version=self.model_ref.version,
            source_annotations=self.model_ref.annotations,
            promotion_metadata={
                "promotion_flow": current.flow_name,
                "promotion_run_id": current.run_id,
            },
        )

        print(f"\nPromoted anomaly_detector v{self.model_ref.version} to CHAMPION")

        # Publish event
        registry.publish_event(self.prj, "model_promoted", payload={
            "model_asset": "anomaly_detector",
            "version": self.model_ref.version,
            "status": "champion",
            "anomaly_rate": self.model_ref.annotations.get("anomaly_rate"),
            "algorithm": self.model_ref.annotations.get("algorithm"),
        })

        self.next(self.end)

    @step
    def end(self):
        """Summary."""
        print(f"\n{'='*50}")
        print("Promotion Complete")
        print(f"{'='*50}")
        print(f"Model: anomaly_detector v{self.model_ref.version}")
        print(f"Status: CHAMPION")
        print(f"\nDeployment configs referencing 'champion' will now use this version.")
        print(f"\nNext steps:")
        print("  1. Deployments using champion alias auto-update")
        print("  2. Or: Update pinned version in deployment config")
        print("  3. Monitor: GET /model/info")


if __name__ == "__main__":
    PromoteDetectorFlow()
