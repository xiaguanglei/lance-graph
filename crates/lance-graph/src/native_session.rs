// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! A lean DataFusion session for plans driven by the native CSR operator.
//!
//! A native plan is a handful of Projection/Filter/Sort/Limit/Aggregate nodes
//! over one operator that already produced a small, single-partition result.
//! The default optimizers run ~25 logical rules for up to three passes and
//! ~20 physical rules, which cost more than the traversal itself. Every
//! optimizer rule only rewrites a plan into an equivalent one, so running
//! fewer of them never changes results; only the rules the physical planner
//! depends on (and the projection rule that keeps unused columns out of the
//! native read) are kept, and `SanityCheckPlan` still rejects any plan whose
//! distribution or ordering requirements are unmet instead of running it.

use std::sync::Arc;

use datafusion::{
    common::tree_node::TreeNode,
    execution::{context::SessionContext, session_state::SessionStateBuilder},
    logical_expr::{Expr, LogicalPlan},
    optimizer::{
        optimize_projections::OptimizeProjections,
        replace_distinct_aggregate::ReplaceDistinctWithAggregate,
    },
    physical_optimizer::{limit_pushdown::LimitPushdown, sanity_checker::SanityCheckPlan},
    prelude::SessionConfig,
};

/// Session for native plans: one partition, lean logical and physical rules.
pub(crate) fn native_session(config: SessionConfig) -> SessionContext {
    let mut config = config.with_target_partitions(1);
    config.options_mut().optimizer.max_passes = 1;
    let state = SessionStateBuilder::new()
        .with_config(config)
        .with_default_features()
        // Distinct must become an Aggregate before physical planning.
        .with_optimizer_rules(vec![
            Arc::new(ReplaceDistinctWithAggregate::new()),
            Arc::new(OptimizeProjections::new()),
        ])
        .with_physical_optimizer_rules(vec![
            Arc::new(LimitPushdown::new()),
            Arc::new(SanityCheckPlan::new()),
        ])
        .build();
    SessionContext::new_with_state(state)
}

/// True when every node of `plan` is one the lean session plans without the
/// rules it leaves out: no joins, unions, windows, subqueries or extensions.
pub(crate) fn lean_plan(plan: &LogicalPlan) -> bool {
    let subquery = |expr: &Expr| {
        expr.exists(|expr| {
            Ok(matches!(
                expr,
                Expr::ScalarSubquery(_)
                    | Expr::Exists(_)
                    | Expr::InSubquery(_)
                    | Expr::SetComparison(_)
                    | Expr::OuterReferenceColumn(..)
            ))
        })
        .unwrap_or(true)
    };
    !plan
        .exists(|node| {
            Ok(!matches!(
                node,
                LogicalPlan::Projection(_)
                    | LogicalPlan::Filter(_)
                    | LogicalPlan::TableScan(_)
                    | LogicalPlan::Sort(_)
                    | LogicalPlan::Limit(_)
                    | LogicalPlan::Aggregate(_)
                    | LogicalPlan::Distinct(_)
                    | LogicalPlan::SubqueryAlias(_)
                    | LogicalPlan::EmptyRelation(_)
            ) || node.expressions().iter().any(subquery))
        })
        .unwrap_or(true)
}
