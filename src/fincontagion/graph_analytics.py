
"""Network analytics for identifying systemically important entities."""

from collections import defaultdict
from decimal import Decimal

import networkx as nx

from fincontagion.domain import FinancialEntity
from fincontagion.instruments import FinancialExposure


ZERO = Decimal("0")


def calculate_systemic_importance(
    entities: list[FinancialEntity],
    exposures: list[FinancialExposure],
) -> list[dict]:
    """Calculate exposure, connectivity, and centrality metrics.

    The original creditor-to-debtor graph represents repayment relationships.
    A separate debtor-to-creditor graph represents potential loss propagation
    when a debtor defaults.
    """

    entity_ids = {entity.entity_id for entity in entities}

    if len(entity_ids) != len(entities):
        raise ValueError("Entity IDs must be unique.")

    financial_graph = nx.DiGraph()
    contagion_graph = nx.DiGraph()

    for entity in entities:
        financial_graph.add_node(entity.entity_id)
        contagion_graph.add_node(entity.entity_id)

    outgoing_amount = defaultdict(lambda: ZERO)
    incoming_amount = defaultdict(lambda: ZERO)
    outgoing_count = defaultdict(int)
    incoming_count = defaultdict(int)
    counterparties = defaultdict(set)
    counterparty_amounts = defaultdict(lambda: defaultdict(lambda: ZERO))

    for exposure in exposures:
        creditor = exposure.creditor_id
        debtor = exposure.debtor_id
        amount = exposure.outstanding_amount

        if creditor not in entity_ids or debtor not in entity_ids:
            raise ValueError(
                f"Exposure {exposure.exposure_id} references "
                "an unknown entity."
            )

        if not amount.is_finite() or amount <= ZERO:
            raise ValueError(
                f"Exposure {exposure.exposure_id} must have a "
                "finite positive amount."
            )

        # Aggregate repeated exposures between the same pair of entities.
        if financial_graph.has_edge(creditor, debtor):
            financial_graph[creditor][debtor]["exposure"] += float(amount)
        else:
            financial_graph.add_edge(
                creditor,
                debtor,
                exposure=float(amount),
            )

        # If a debtor defaults, the creditor may incur a loss.
        if contagion_graph.has_edge(debtor, creditor):
            contagion_graph[debtor][creditor]["exposure"] += float(amount)
        else:
            contagion_graph.add_edge(
                debtor,
                creditor,
                exposure=float(amount),
            )

        outgoing_amount[creditor] += amount
        incoming_amount[debtor] += amount
        outgoing_count[creditor] += 1
        incoming_count[debtor] += 1

        counterparties[creditor].add(debtor)
        counterparties[debtor].add(creditor)

        counterparty_amounts[creditor][debtor] += amount
        counterparty_amounts[debtor][creditor] += amount

    # Centrality measures describe the structure of the network.
    degree_centrality = nx.degree_centrality(financial_graph)

    # Betweenness is calculated on an unweighted graph. It measures
    # structural bridging, not the monetary size of an exposure.
    betweenness_centrality = nx.betweenness_centrality(
        financial_graph,
        normalized=True,
    )

    # PageRank is a recursive connectivity measure, not a failure probability.
    pagerank = nx.pagerank(
        financial_graph,
        alpha=0.85,
        weight="exposure",
    )

    # A debtor-to-creditor view helps inspect potential loss-propagation paths.
    contagion_degree_centrality = nx.degree_centrality(contagion_graph)
    contagion_betweenness = nx.betweenness_centrality(
        contagion_graph,
        normalized=True,
    )

    results = []

    for entity in entities:
        entity_id = entity.entity_id
        total_outgoing = outgoing_amount[entity_id]
        total_incoming = incoming_amount[entity_id]
        total_exposure = total_outgoing + total_incoming

        largest_counterparty_exposure = max(
            counterparty_amounts[entity_id].values(),
            default=ZERO,
        )

        concentration_ratio = (
            largest_counterparty_exposure / total_exposure
            if total_exposure > ZERO
            else ZERO
        )

        results.append(
            {
                "entity_id": entity_id,
                "entity_type": entity.entity_type.value,
                "outgoing_exposure": total_outgoing,
                "incoming_exposure": total_incoming,
                "total_connected_exposure": total_exposure,
                "outgoing_exposure_count": outgoing_count[entity_id],
                "incoming_exposure_count": incoming_count[entity_id],
                "unique_counterparties": len(counterparties[entity_id]),
                "largest_counterparty_exposure": (
                    largest_counterparty_exposure
                ),
                "largest_counterparty_concentration": concentration_ratio,
                "degree_centrality": degree_centrality[entity_id],
                "betweenness_centrality": betweenness_centrality[entity_id],
                "pagerank": pagerank[entity_id],
                "contagion_degree_centrality": (
                    contagion_degree_centrality[entity_id]
                ),
                "contagion_betweenness_centrality": (
                    contagion_betweenness[entity_id]
                ),
            }
        )

    results.sort(
        key=lambda row: (
            row["total_connected_exposure"],
            row["unique_counterparties"],
        ),
        reverse=True,
    )

    for rank, row in enumerate(results, start=1):
        row["exposure_importance_rank"] = rank

    # Separate ranking: structural bridging potential in the financial graph.
    betweenness_order = sorted(
        results,
        key=lambda row: row["betweenness_centrality"],
        reverse=True,
    )

    for rank, row in enumerate(betweenness_order, start=1):
        row["betweenness_rank"] = rank

    return results
