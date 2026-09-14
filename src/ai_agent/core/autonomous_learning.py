"""Bounded autonomous learning orchestration with conservative source comparison."""
from __future__ import annotations

from dataclasses import dataclass, field

from .conflict_resolution import ConflictAnalysis, ConflictResolver
from .experience import Experience, ExperienceStore
from .invariants import assert_core_invariants
from .knowledge import KnowledgeItem
from .learning import LearningEngine
from .learning_planner import LearningPlanner, LearningTask
from .research_plan import ResearchPlan, ResearchPlanner
from .researcher import InternetResearcher, ResearchDocument
from .source_comparison import SourceComparator, SourceComparison


@dataclass
class LearningAttempt:
    task_id: str
    objective: str
    status: str
    reason: str
    knowledge_id: str | None = None
    source_uri: str | None = None


@dataclass
class AutonomousLearningSession:
    goal: str
    attempts: list[LearningAttempt] = field(default_factory=list)
    comparison: SourceComparison | None = None
    conflict_analysis: ConflictAnalysis | None = None

    @property
    def completed(self) -> bool:
        return any(attempt.status == "verified" for attempt in self.attempts)


class AutonomousLearningCoordinator:
    """Connect planning to discovery, research, comparison, conflict detection, verification, and experience."""

    def __init__(self, *, planner: LearningPlanner | None = None,
                 research_planner: ResearchPlanner | None = None,
                 researcher: InternetResearcher | None = None,
                 learner: LearningEngine | None = None,
                 comparator: SourceComparator | None = None,
                 conflict_resolver: ConflictResolver | None = None,
                 experience_store: ExperienceStore | None = None,
                 max_attempts_per_task: int = 3) -> None:
        if max_attempts_per_task <= 0:
            raise ValueError("max_attempts_per_task must be positive")
        self.planner = planner or LearningPlanner()
        self.research_planner = research_planner
        self.researcher = researcher or InternetResearcher()
        self.learner = learner or LearningEngine()
        self.comparator = comparator or SourceComparator()
        self.conflict_resolver = conflict_resolver or ConflictResolver()
        self.experience_store = experience_store or ExperienceStore()
        self.max_attempts_per_task = max_attempts_per_task

    def plan(self, goal: str) -> list[LearningTask]:
        assert_core_invariants()
        return self.planner.plan(goal, self.learner.knowledge)

    def discover(self, learning_task: LearningTask) -> ResearchPlan:
        assert_core_invariants()
        if self.research_planner is None:
            raise RuntimeError("research_planner is required for autonomous source discovery")
        return self.research_planner.plan(learning_task.objective)

    def research(self, learning_task: LearningTask, uri: str) -> tuple[ResearchDocument, KnowledgeItem]:
        assert_core_invariants()
        document = self.researcher.fetch(uri)
        item = self.learner.observe(learning_task.objective, document, kind="lesson",
                                    tags=["autonomous-learning", learning_task.gap_id])
        return document, item

    def verify(self, learning_task: LearningTask, item: KnowledgeItem, evidence: list[str]) -> LearningAttempt:
        assert_core_invariants()
        self.learner.verify(item, evidence)
        return LearningAttempt(learning_task.gap_id, learning_task.objective, "verified",
                               "Evidence and provenance accepted.", item.id,
                               item.sources[0].uri if item.sources else None)

    def _record_experience(self, goal: str, session: AutonomousLearningSession) -> None:
        assert_core_invariants()
        if session.completed:
            self.experience_store.record(Experience(
                goal=goal, outcome="verified",
                lesson="Verified learning outcome with provenance and evidence.", success=True,
                evidence=[a.reason for a in session.attempts if a.status == "verified"],
                tags=["autonomous-learning", "success"],
            ))
        else:
            failed = [a for a in session.attempts if a.status in {"failed", "conflicted"}]
            if failed:
                self.experience_store.record(Experience(
                    goal=goal, outcome=failed[-1].status, lesson=failed[-1].reason, success=False,
                    evidence=[a.reason for a in failed], tags=["autonomous-learning", "failure"],
                ))

    def discover_and_research(self, goal: str, evidence: list[str] | None = None) -> AutonomousLearningSession:
        """Search, fetch, compare, detect conflicts, and conservatively verify one learning gap."""
        assert_core_invariants()
        session = AutonomousLearningSession(goal=goal)
        tasks = self.plan(goal)
        if not tasks:
            session.attempts.append(LearningAttempt("", goal, "no_gap", "No learning gap was detected."))
            self._record_experience(goal, session)
            return session
        task = tasks[0]
        try:
            research_plan = self.discover(task)
            documents: list[ResearchDocument] = []
            items: list[KnowledgeItem] = []
            for source in research_plan.sources:
                try:
                    document, item = self.research(task, source.url)
                    documents.append(document)
                    items.append(item)
                    session.attempts.append(LearningAttempt(task.gap_id, task.objective, "proposed",
                                                            "Source retrieved; comparison, conflict detection, and verification pending.",
                                                            item.id, source.url))
                except Exception as exc:
                    session.attempts.append(LearningAttempt(task.gap_id, task.objective, "failed", str(exc), source_uri=source.url))

            session.conflict_analysis = self.conflict_resolver.analyze(goal, documents)
            if session.conflict_analysis.conflicted:
                item_ids = {item.id for item in items}
                for item in items:
                    self.learner.conflict(item, session.conflict_analysis.reason)
                for attempt in session.attempts:
                    if attempt.knowledge_id in item_ids and attempt.status == "proposed":
                        attempt.status = "conflicted"
                        attempt.reason = session.conflict_analysis.reason
                self._record_experience(goal, session)
                return session

            if evidence and items:
                self.learner.verify(items[0], evidence)
                for attempt in session.attempts:
                    if attempt.knowledge_id == items[0].id:
                        attempt.status = "verified"
                        attempt.reason = "Explicit evidence and source provenance accepted."
                        break
            else:
                session.comparison = self.comparator.compare(goal, documents)
                if session.comparison.corroborated and items:
                    self.learner.verify(items[0], session.comparison.evidence)
                    for attempt in session.attempts:
                        if attempt.knowledge_id == items[0].id:
                            attempt.status = "verified"
                            attempt.reason = session.comparison.reason
                            break
        except Exception as exc:
            session.attempts.append(LearningAttempt(task.gap_id, task.objective, "failed", str(exc)))
        self._record_experience(goal, session)
        return session

    def run_once(self, goal: str, uri: str | None = None,
                 evidence: list[str] | None = None) -> AutonomousLearningSession:
        assert_core_invariants()
        if uri is None:
            return self.discover_and_research(goal, evidence=evidence)
        session = AutonomousLearningSession(goal=goal)
        tasks = self.plan(goal)
        if not tasks:
            session.attempts.append(LearningAttempt("", goal, "no_gap", "No learning gap was detected."))
            self._record_experience(goal, session)
            return session
        task = tasks[0]
        try:
            _, item = self.research(task, uri)
            if evidence is None:
                session.attempts.append(LearningAttempt(task.gap_id, task.objective, "proposed",
                                                        "Source retrieved; verification still required.", item.id, uri))
            else:
                session.attempts.append(self.verify(task, item, evidence))
        except Exception as exc:
            session.attempts.append(LearningAttempt(task.gap_id, task.objective, "failed", str(exc)))
        self._record_experience(goal, session)
        return session
