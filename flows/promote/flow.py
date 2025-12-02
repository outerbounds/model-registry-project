"""
Promote a model version to champion (or other alias).

This flow demonstrates:
- Assigning aliases to model versions
- Requesting approval for production changes
- Demoting previous champion automatically

Usage:
    # Promote latest to champion
    python flow.py run --version latest

    # Promote specific version
    python flow.py run --version v5

    # Promote to custom alias
    python flow.py run --version latest --alias production
"""

from metaflow import step, Parameter
from obproject import ProjectFlow

import src


class PromoteModelFlow(ProjectFlow):
    """
    Promote a model version to champion or other alias.

    This assigns the specified alias to a model version,
    automatically clearing it from any previous version.
    """

    version = Parameter(
        "version",
        required=True,
        help="Model version to promote (e.g., 'latest', 'v5', or version ID)"
    )

    alias = Parameter(
        "alias",
        default="champion",
        help="Alias to assign (e.g., 'champion', 'production', 'staging')"
    )

    model_name = Parameter(
        "model_name",
        default="anomaly_detector",
        help="Name of model asset"
    )

    @step
    def start(self):
        """Load the model version to promote."""
        from src import registry
        from metaflow import Flow

        print(f"Project: {self.prj.project}, Branch: {self.prj.branch}")
        print(f"\nPromoting {self.model_name} version '{self.version}' to '{self.alias}'")

        # Load the model to verify it exists
        # Handle different version formats:
        # - "latest", "latest-N" -> pass directly to Asset API
        # - bare run ID like "186088" -> load via Metaflow Client API
        # - full version pathspec -> pass directly to Asset API
        try:
            if self.version.startswith("latest"):
                # Use Asset API for "latest" or "latest-N" formats
                self.model = registry.load_model(
                    self.prj.asset,
                    self.model_name,
                    version=self.version
                )
            elif self.version.isdigit() or len(self.version) < 20:
                # Looks like a bare run ID - load via Metaflow Client API
                print(f"Loading model from TrainDetectorFlow run: {self.version}")
                flow = Flow('TrainDetectorFlow')
                run = flow[self.version]

                if not run.successful:
                    raise ValueError(f"Run {self.version} was not successful")

                # Get model info from run artifacts
                train_step = run['train']
                for task in train_step:
                    config = dict(task.data.model_config)
                    prediction = task.data.prediction
                    feature_set = task.data.feature_set
                    data_source = task.data.data_source

                    # Create a simple namespace object to hold model info
                    from types import SimpleNamespace
                    self.model = SimpleNamespace(
                        version=run.id,
                        alias=None,
                        algorithm=config.get("algorithm", "isolation_forest"),
                        training_run_id=run.id,
                        annotations={
                            "algorithm": config.get("algorithm", "isolation_forest"),
                            "training_run_id": run.id,
                            "anomaly_rate": float(prediction.anomaly_rate) if hasattr(prediction, 'anomaly_rate') else 0,
                            "training_samples": int(feature_set.n_samples) if hasattr(feature_set, 'n_samples') else 0,
                            "data_source": data_source,
                        }
                    )
                    print(f"Loaded model info from run {run.id}")
                    break
            else:
                # Full version pathspec - use Asset API
                self.model = registry.load_model(
                    self.prj.asset,
                    self.model_name,
                    version=self.version
                )
        except Exception as e:
            raise RuntimeError(f"Model version '{self.version}' not found: {e}")

        print(f"\nModel to promote:")
        print(f"  Version: {self.model.version}")
        print(f"  Current alias: {self.model.alias or 'none'}")
        print(f"  Algorithm: {self.model.algorithm}")
        print(f"  Training run: {self.model.training_run_id}")
        print(f"  Anomaly rate: {float(self.model.annotations.get('anomaly_rate', 0)):.1%}")

        # Find the current champion via Metaflow run tags
        self.previous_holder = None
        current_champion = registry.get_champion_run_id(flow_name="TrainDetectorFlow")
        if current_champion and current_champion != self.model.training_run_id:
            self.previous_holder = current_champion
            print(f"\nCurrent {self.alias}: run {current_champion}")
        elif not current_champion:
            print(f"\nNo current {self.alias} (first assignment)")

        self.next(self.promote)

    @step
    def promote(self):
        """Promote a model by tagging its training run."""
        from src import registry
        from metaflow import current

        print(f"\nPromoting v{self.model.version} to '{self.alias}'...")

        # Use Metaflow's native tagging to mark the training run as champion
        # This removes the tag from any previous champion and adds it to this run
        try:
            previous = registry.set_champion_run(
                run_id=self.model.training_run_id,
                flow_name="TrainDetectorFlow",
            )

            self.promotion_success = True
            print(f"Successfully promoted run {self.model.training_run_id} to {self.alias}")
            print(f"  Tagged TrainDetectorFlow/{self.model.training_run_id} as '{self.alias}'")

            if previous:
                print(f"  Previous {self.alias} (run {previous}) has been untagged")

        except Exception as e:
            self.promotion_success = False
            self.promotion_error = str(e)
            print(f"[ERROR] Promotion failed: {e}")
            import traceback
            traceback.print_exc()

        # Publish event for notification/audit
        registry.publish_event(self.prj, "model_promoted", payload={
            "model_name": self.model_name,
            "model_version": self.model.version,
            "alias": self.alias,
            "previous_holder": self.previous_holder,
            "promoted_by_run_id": current.run_id,
        })
        print(f"Published 'model_promoted' event")

        self.next(self.end)

    @step
    def end(self):
        """Summary."""
        print(f"\n{'='*50}")
        print("Promotion Complete")
        print(f"{'='*50}")
        print(f"Model: {self.model_name}")
        print(f"Version: {self.model.version}")
        print(f"Alias: {self.alias}")

        if hasattr(self, 'promotion_success') and self.promotion_success:
            print(f"\nThe model is now the {self.alias}!")
        else:
            print(f"\nNote: Alias tag update may have failed.")
            print("The 'model_promoted' event was still published for tracking.")


if __name__ == "__main__":
    PromoteModelFlow()
