import torch
from qiskit import QuantumCircuit
from qiskit.circuit.library import ZZFeatureMap, EfficientSU2
from qiskit.primitives import Estimator
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.neural_networks import EstimatorQNN
from torch import nn


class QuantumDigitClassifier(nn.Module):
    def __init__(
            self,
            num_qubits: int = 4,
            num_classes: int = 10,
            input_dim: int = 16,
    ):
        """
        Quantum classifier:
        1. Classical frontend compresses 28x28 -> input_dim
        2. Quantum circuit applied to input_dim features via feature map + ansatz
        3. QNN outputs single expectation value
        4. Linear layer maps to num_classes classes [digit 0-9]
        """
        super().__init__()

        # Classical feature extractor: image -> input_dim
        self.feature_extractor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, input_dim),
            nn.Tanh(),  # keep features in [-1, 1]
        )

        # feature map + ansatz
        feature_map = ZZFeatureMap(num_qubits, reps=1)
        ansatz = EfficientSU2(num_qubits, reps=2, entanglement="linear")

        qc = QuantumCircuit(num_qubits)
        qc.compose(feature_map, inplace=True)
        qc.compose(ansatz, inplace=True)

        estimator = Estimator()

        qnn = EstimatorQNN(
            circuit=qc,
            estimator=estimator,
            input_params=feature_map.parameters,  # corresponds to classical features
            weight_params=ansatz.parameters,  # trainable parameters
        )

        # wrap QNN in TorchConnector for pytorch
        self.q_layer = TorchConnector(qnn)

        # map qnn scalar output to num_classes
        self.classifier = nn.Linear(1, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = next(self.parameters()).device
        x = x.to(device)
        x = self.feature_extractor(x)  # [batch, input_dim]

        # q_layer runs via Qiskit backend
        q_out = self.q_layer(x)  # [batch, 1]
        logits = self.classifier(q_out)
        return logits
