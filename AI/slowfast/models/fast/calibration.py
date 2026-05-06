import math

import torch
from torch import nn


def binary_entropy(probability, eps=1e-6):
    p = min(max(float(probability), eps), 1.0 - eps)
    return float(-(p * math.log(p) + (1.0 - p) * math.log(1.0 - p)))


def expected_calibration_error(probabilities, labels, bins=10):
    if not probabilities:
        return 0.0
    bin_edges = torch.linspace(0.0, 1.0, steps=bins + 1)
    probs = torch.tensor(probabilities, dtype=torch.float32)
    targets = torch.tensor(labels, dtype=torch.float32)
    predictions = (probs >= 0.5).to(torch.float32)
    ece = 0.0

    for left, right in zip(bin_edges[:-1], bin_edges[1:]):
        if right >= 1.0:
            mask = (probs >= left) & (probs <= right)
        else:
            mask = (probs >= left) & (probs < right)
        if not torch.any(mask):
            continue
        bin_probs = probs[mask]
        bin_targets = targets[mask]
        bin_predictions = predictions[mask]
        confidence = torch.mean(torch.where(bin_predictions > 0, bin_probs, 1.0 - bin_probs))
        accuracy = torch.mean((bin_predictions == bin_targets).to(torch.float32))
        ece += float(torch.abs(confidence - accuracy) * (mask.sum().item() / len(probabilities)))
    return float(ece)


def summarize_probabilities(probabilities):
    if not probabilities:
        return {"count": 0}
    values = sorted(float(value) for value in probabilities)

    def pick(ratio):
        index = int(round((len(values) - 1) * ratio))
        return values[index]

    return {
        "count": len(values),
        "mean": float(sum(values) / len(values)),
        "min": values[0],
        "p25": pick(0.25),
        "p50": pick(0.50),
        "p75": pick(0.75),
        "p90": pick(0.90),
        "p95": pick(0.95),
        "max": values[-1],
    }


def evaluate_logits(logits, labels, temperature=1.0):
    logits = torch.as_tensor(logits, dtype=torch.float32)
    labels = torch.as_tensor(labels, dtype=torch.float32)
    scaled_logits = logits / max(float(temperature), 1e-3)
    probabilities = torch.sigmoid(scaled_logits)

    bce = nn.BCEWithLogitsLoss()(scaled_logits, labels)
    brier = torch.mean((probabilities - labels) ** 2)
    accuracy = torch.mean(((probabilities >= 0.5).to(torch.float32) == labels).to(torch.float32))

    positive_probs = probabilities[labels > 0.5].tolist()
    negative_probs = probabilities[labels <= 0.5].tolist()

    return {
        "temperature": float(temperature),
        "bce": float(bce.item()),
        "brier": float(brier.item()),
        "accuracy": float(accuracy.item()),
        "ece": expected_calibration_error(probabilities.tolist(), labels.tolist()),
        "positive_prob_summary": summarize_probabilities(positive_probs),
        "negative_prob_summary": summarize_probabilities(negative_probs),
        "positive_entropy_mean": _mean_entropy(positive_probs),
        "negative_entropy_mean": _mean_entropy(negative_probs),
    }


def _mean_entropy(probabilities):
    if not probabilities:
        return 0.0
    return float(sum(binary_entropy(value) for value in probabilities) / len(probabilities))


def fit_temperature(logits, labels, init_temperature=1.0, max_iter=50):
    logits = torch.as_tensor(logits, dtype=torch.float32)
    labels = torch.as_tensor(labels, dtype=torch.float32)
    log_temperature = torch.tensor([math.log(max(float(init_temperature), 1e-3))], requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=max_iter, line_search_fn="strong_wolfe")
    criterion = nn.BCEWithLogitsLoss()

    def closure():
        optimizer.zero_grad()
        temperature = torch.exp(log_temperature)
        loss = criterion(logits / temperature, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    fitted_temperature = float(torch.exp(log_temperature).detach().item())
    return max(fitted_temperature, 1e-3)
