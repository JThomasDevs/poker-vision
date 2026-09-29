"""
Poker hand evaluation engine using treys.
"""

from treys import Evaluator, Deck as TreysDeck, Card
from typing import List, Tuple, Optional, Sequence, TYPE_CHECKING
from dataclasses import dataclass
import random

if TYPE_CHECKING:
    from src.detection.table_amounts import TableAmounts


# Rank ints from treys: 0=2 .. 12=A
_BROADWAY = frozenset(range(8, 13))  # T–A
_VALID_RANGES = frozenset({"random", "strong", "default", "value", "nuts"})

# Pot-odds call pricing (facing a bet). Documented thresholds for recommend().
_CALL_MARGIN = 0.02          # prefer call when win_prob >= pot_odds + this
_RAISE_OVER_ODDS = 0.15      # raise when win_prob >= pot_odds + this
_RAISE_ABS_EQUITY = 0.65     # or when absolute equity is clearly high


@dataclass
class HandResult:
    """Result of hand evaluation."""
    hand_type: str  # "High Card", "Pair", "Two Pair", etc.
    hand_rank: int  # 1-7462 (lower is better)
    win_probability: float  # 0.0 - 1.0
    recommendation: str  # "fold", "call", "check/fold", "check/call", "bet", "raise"
    ev: float  # Expected value
    pot_odds: Optional[float] = None  # set when facing a bet and amounts known
    
    def __str__(self):
        return f"{self.hand_type} (rank {self.hand_rank}) - {self.win_probability:.1%} win - {self.recommendation}"


class PokerEngine:
    """Evaluates poker hands and provides recommendations.

    When ``TableAmounts.to_call > 0``, actions use pot-odds pricing
    (``recommend``). Otherwise the win_prob ladder (check/fold, check/call,
    bet, …) applies. Equity is Monte Carlo vs a villain sampling mode —
    see ``calculate_equity`` / ``villain_range``.
    """
    
    def __init__(self, n_simulations: int = 1000, villain_range: str = "strong"):
        """Initialize engine.
        
        Args:
            n_simulations: Monte Carlo simulations for equity calc
            villain_range: Default opponent sampling mode
                (``random`` | ``strong``/``default`` | ``value``/``nuts``).
                Live app defaults to ``strong`` so win% reflects tougher
                opposition than uniform random two cards.
        """
        self.n_simulations = n_simulations
        self.villain_range = self._normalize_range(villain_range)
        self.evaluator = Evaluator()

    @staticmethod
    def _normalize_range(villain_range: Optional[str]) -> str:
        """Map aliases; unknown/empty → strong (not random)."""
        mode = (villain_range or "strong").strip().lower()
        if mode in ("", "default"):
            return "strong"
        if mode == "nuts":
            return "value"
        if mode not in _VALID_RANGES:
            return "strong"
        return mode
    
    @staticmethod
    def normalize_card(card: str) -> str:
        """Normalize to treys form: rank in 2-9TJQKA + suit in hdcs (e.g. As, Td)."""
        c = (card or "").strip().replace("10", "T")
        if len(c) < 2:
            raise ValueError(f"bad card: {card!r}")
        rank, suit = c[0].upper(), c[1].lower()
        if rank == "1" and len(c) >= 3 and c[1] == "0":
            rank, suit = "T", c[2].lower()
        if rank not in "23456789TJQKA" or suit not in "hdcs":
            raise ValueError(f"bad card: {card!r}")
        return f"{rank}{suit}"

    def cards_to_treys(self, cards: List[str]) -> List:
        """Convert card strings to treys Card objects.
        
        Args:
            cards: List like ["Ah", "Kd", "Tc"]
            
        Returns:
            List of treys.Card objects
        """
        return [Card.new(self.normalize_card(c)) for c in cards]
    
    def evaluate_hand(
        self, 
        hole_cards: List[str], 
        community_cards: List[str],
        villain_range: Optional[str] = None,
        amounts: Optional["TableAmounts"] = None,
    ) -> HandResult:
        """Evaluate a poker hand.
        
        Args:
            hole_cards: Player's two cards ["As", "Kh"]
            community_cards: Community cards on table ["Tc", "Jd", "2s"]
            villain_range: Opponent sampling mode used for equity (and thus
                recommendations). Modes:
                  - ``random``: uniform two cards from the remaining deck
                    (explicit opt-in; old behavior).
                  - ``strong`` / ``default``: weighted toward pairs, broadway,
                    suited connectors, and board-connecting hands; trash is
                    heavily downweighted. **Default** for the live app.
                  - ``value`` / ``nuts``: only hands that beat or tie hero on
                    the known board (≥3 cards); falls back to ``strong`` when
                    the board is incomplete. Still uses full treys evaluate
                    (kickers matter) once the board is completed in sim.
            amounts: Optional pot / to_call / hero_stack for pot-odds pricing.
            
        Returns:
            HandResult with evaluation and recommendation
        """
        if len(hole_cards) != 2:
            return HandResult("Invalid", 7462, 0.0, "error", 0.0)

        try:
            norm_hole = [self.normalize_card(c) for c in hole_cards]
            norm_board = [self.normalize_card(c) for c in community_cards]
        except ValueError:
            return HandResult("Invalid cards", 7462, 0.0, "error", 0.0)

        # Duplicates (common when CNN mislabels blank/mock blobs) break treys lookups.
        all_norm = norm_hole + norm_board
        if len(set(all_norm)) != len(all_norm):
            return HandResult("Duplicate cards", 7462, 0.0, "error", 0.0)
        
        # Convert to treys format
        try:
            hand = self.cards_to_treys(norm_hole)
            board = self.cards_to_treys(norm_board)
        except Exception:
            return HandResult("Invalid cards", 7462, 0.0, "error", 0.0)
        
        # Get hand type and rank (treys needs 0 or 3–5 board cards)
        if len(board) >= 3:
            try:
                eval_rank = self.evaluator.evaluate(hand, board)
                hand_type = self.evaluator.class_to_string(self.evaluator.get_rank_class(eval_rank))
            except Exception:
                return HandResult("Eval error", 7462, 0.0, "error", 0.0)
        else:
            # Pre-flop / incomplete board — skip showdown rank
            eval_rank = 7462
            hand_type = "Pre-flop" if len(board) == 0 else "Incomplete board"
        
        mode = self._normalize_range(
            villain_range if villain_range is not None else self.villain_range
        )
        # Calculate win probability via Monte Carlo (recommendations use this)
        win_prob = self.calculate_equity(hand, board, villain_range=mode)
        
        # Action: pot-odds when facing a bet; else win_prob ladder
        recommendation, odds = self.recommend(win_prob, amounts=amounts, n_board=len(board))
        
        # Calculate EV (simplified)
        ev = self.calculate_ev(win_prob, recommendation)
        
        return HandResult(
            hand_type=hand_type,
            hand_rank=eval_rank,
            win_probability=win_prob,
            recommendation=recommendation,
            ev=ev,
            pot_odds=odds,
        )

    @staticmethod
    def _all_combos(cards: Sequence[int]) -> List[Tuple[int, int]]:
        """Unordered 2-card combos from remaining deck ints."""
        out: List[Tuple[int, int]] = []
        n = len(cards)
        for i in range(n):
            for j in range(i + 1, n):
                out.append((cards[i], cards[j]))
        return out

    def _strong_weight(self, c1: int, c2: int, board: Sequence[int]) -> float:
        """Weight a villain hole pair for ``strong`` sampling.

        Overweights pairs, broadway, suited connectors, and board hits;
        heavily downweights unconnected low trash that misses the board.
        """
        r1, r2 = Card.get_rank_int(c1), Card.get_rank_int(c2)
        s1, s2 = Card.get_suit_int(c1), Card.get_suit_int(c2)
        high, low = max(r1, r2), min(r1, r2)
        paired = r1 == r2
        suited = s1 == s2
        gap = high - low

        board_ranks = [Card.get_rank_int(c) for c in board]
        board_suits = [Card.get_suit_int(c) for c in board]
        hits_board = (r1 in board_ranks) or (r2 in board_ranks)

        w = 1.0

        if paired:
            # Pocket pairs: higher pairs much more likely vs a continuing range
            w += 10.0 + high * 1.5
        else:
            if high in _BROADWAY:
                w += 4.0
            if low in _BROADWAY:
                w += 2.5
            if suited:
                w += 2.0
                if gap <= 2:
                    w += 3.5  # suited connector / one-gap
                elif gap <= 4 and high in _BROADWAY:
                    w += 1.5
            elif gap <= 1 and high >= 8:
                w += 1.5  # offsuit broadway connector

        if board:
            # Pair / two pair / set potential on this board
            if hits_board:
                w += 8.0
                top_board = max(board_ranks)
                if r1 == top_board or r2 == top_board:
                    w += 4.0  # top pair territory
                # Set when pocket matches a board rank is already covered by pair+hit
                if paired and r1 in board_ranks:
                    w += 12.0

            # Flush-draw-ish: suited hole matching a board suit with ≥2 of that suit
            if suited and board_suits.count(s1) >= 2:
                w += 5.0
            elif suited and board_suits.count(s1) == 1:
                w += 1.5

            # Straight-ish: hole ranks near board ranks
            if board_ranks and not paired:
                for br in board_ranks:
                    if abs(r1 - br) <= 2 or abs(r2 - br) <= 2:
                        w += 1.5
                        break

        # Trash: both low, disconnected, miss board
        if (
            not paired
            and high < 8
            and gap > 3
            and not hits_board
            and not (suited and gap <= 2)
        ):
            w *= 0.12

        return max(w, 0.01)

    def _value_mask(
        self,
        combos: List[Tuple[int, int]],
        hand: List[int],
        board: List[int],
    ) -> List[bool]:
        """True where villain beats or ties hero on the known board (≥3 cards)."""
        hero_rank = self.evaluator.evaluate(hand, board)
        mask: List[bool] = []
        for c1, c2 in combos:
            try:
                opp_rank = self.evaluator.evaluate([c1, c2], board)
                mask.append(opp_rank <= hero_rank)  # lower = better
            except Exception:
                mask.append(False)
        return mask

    def _weighted_pick(
        self,
        combos: List[Tuple[int, int]],
        weights: List[float],
    ) -> Optional[Tuple[int, int]]:
        if not combos:
            return None
        total = sum(weights)
        if total <= 0:
            return random.choice(combos)
        # random.choices keeps kickers/full evaluate path — only changes sampling
        return random.choices(combos, weights=weights, k=1)[0]
    
    def calculate_equity(
        self,
        hand: List,
        board: List,
        villain_range: Optional[str] = None,
    ) -> float:
        """Calculate win probability via Monte Carlo simulation.

        ``villain_range`` controls how opponent hole cards are sampled
        (recommendations use this equity unchanged except for sampling):

        - ``random``: uniform from remaining deck (legacy / opt-in).
        - ``strong`` / ``default``: weighted toward plausible strong hands
          given the board (pairs, broadway, suited connectors, board hits);
          trash is downweighted. Default for new calls.
        - ``value`` / ``nuts``: only hands that beat or tie hero on the known
          board when ≥3 community cards are present; otherwise same as
          ``strong``. Board completion and showdown still use full treys
          evaluate, so kickers matter.
        """
        mode = self._normalize_range(
            villain_range if villain_range is not None else self.villain_range
        )

        known = list(hand) + list(board)
        base_deck = TreysDeck()
        for card in known:
            try:
                base_deck.cards.remove(card)
            except ValueError:
                pass
        available = list(base_deck.cards)
        if len(available) < 2:
            return 0.0

        # Precompute villain sampling distribution once (same known cards each trial)
        combos = self._all_combos(available)
        sample_combos = combos
        sample_weights: Optional[List[float]] = None
        if mode == "random":
            sample_combos = combos
            sample_weights = None
        elif mode == "value" and len(board) >= 3:
            mask = self._value_mask(combos, list(hand), list(board))
            value_combos = [c for c, ok in zip(combos, mask) if ok]
            if value_combos:
                sample_combos = value_combos
                sample_weights = [
                    self._strong_weight(c1, c2, board) for c1, c2 in value_combos
                ]
            else:
                sample_weights = [
                    self._strong_weight(c1, c2, board) for c1, c2 in combos
                ]
        else:
            # strong, or value with incomplete board
            sample_weights = [
                self._strong_weight(c1, c2, board) for c1, c2 in combos
            ]

        wins = 0.0
        trials = 0
        hand_l = list(hand)
        board_l = list(board)

        for _ in range(self.n_simulations):
            if sample_weights is None:
                opp = random.choice(sample_combos)
            else:
                opp = self._weighted_pick(sample_combos, sample_weights)
            if opp is None:
                continue
            opp_hand = list(opp)
            opp_set = set(opp_hand)

            rem = [c for c in available if c not in opp_set]
            remaining_board = list(board_l)
            needed = 5 - len(remaining_board)
            if needed > 0:
                if len(rem) < needed:
                    continue
                remaining_board.extend(random.sample(rem, needed))

            # Evaluate — full treys path so kickers still matter
            try:
                our_rank = self.evaluator.evaluate(hand_l, remaining_board)
                opp_rank = self.evaluator.evaluate(opp_hand, remaining_board)
                if our_rank < opp_rank:  # Lower rank = better hand
                    wins += 1
                elif our_rank == opp_rank:
                    wins += 0.5  # Chop
                trials += 1
            except Exception:
                continue

        return wins / trials if trials > 0 else 0.0
    
    def recommend(
        self,
        win_prob: float,
        amounts: Optional["TableAmounts"] = None,
        n_board: int = 5,
    ) -> Tuple[str, Optional[float]]:
        """Choose an action from equity and optional table amounts.

        Facing a bet (``to_call > 0``):
          pot_odds = to_call / (pot + to_call)  (pot defaults to 0 if unknown)
          call  when win_prob >= pot_odds + 0.02
          fold  when below that
          raise when win_prob >= pot_odds + 0.15 or win_prob >= 0.65
          If ``to_call >= hero_stack``, treat as all-in: only call or fold
          (no raise).

        Checked to us (``to_call == 0``) or amounts unknown: win_prob ladder
        via ``get_recommendation`` (check/fold, check/call, bet, raise, …).

        Returns:
            (action, pot_odds_or_None)
        """
        to_call = getattr(amounts, "to_call", None) if amounts is not None else None
        pot = getattr(amounts, "pot", None) if amounts is not None else None
        hero_stack = getattr(amounts, "hero_stack", None) if amounts is not None else None

        if to_call is not None and to_call > 0:
            pot_v = float(pot) if pot is not None and pot > 0 else 0.0
            odds = to_call / (pot_v + to_call)
            all_in = hero_stack is not None and to_call >= hero_stack * 0.99

            if (
                not all_in
                and (
                    win_prob >= odds + _RAISE_OVER_ODDS
                    or win_prob >= _RAISE_ABS_EQUITY
                )
            ):
                return "raise", odds
            if win_prob >= odds + _CALL_MARGIN:
                return "call", odds
            return "fold", odds

        # to_call == 0 (checked) or unreadable → existing equity ladder
        return self.get_recommendation(win_prob, n_board=n_board), None

    def get_recommendation(self, win_prob: float, n_board: int = 5) -> str:
        """Map equity to a coarse action when not facing a priced bet.

        Used when ``to_call`` is 0 or unknown. ``n_board`` softens thresholds
        early (more board unknown → slightly more cautious).

        Threshold bands (postflop / complete board; early streets similar):
          raise      >= raise_at (~0.70–0.75)
          bet        >= bet_at (~0.55)
          check/call >= check_call_at (midpoint of check_at..bet_at, ~0.48)
          check/fold >= check_at (~0.40–0.42)
          call       >= call_at (~0.25–0.28)
          fold       below call_at

        check/call = checking is fine and calling a bet is still reasonable
        (stronger mid-equity). check/fold = checking is OK but calling is not
        worth it (weaker mid-equity).
        """
        # Slightly higher bar preflop / on incomplete boards (more variance).
        if n_board <= 0:
            raise_at, bet_at, check_at, call_at = 0.70, 0.55, 0.42, 0.28
        elif n_board < 3:
            raise_at, bet_at, check_at, call_at = 0.72, 0.55, 0.40, 0.26
        elif n_board < 5:
            raise_at, bet_at, check_at, call_at = 0.72, 0.55, 0.40, 0.25
        else:
            raise_at, bet_at, check_at, call_at = 0.75, 0.55, 0.40, 0.25

        # Split former "check" band: stronger mid → check/call, weaker → check/fold.
        check_call_at = (check_at + bet_at) / 2.0

        if win_prob >= raise_at:
            return "raise"
        if win_prob >= bet_at:
            return "bet"
        if win_prob >= check_call_at:
            return "check/call"
        if win_prob >= check_at:
            return "check/fold"
        if win_prob >= call_at:
            return "call"
        return "fold"
    
    def calculate_ev(self, win_prob: float, recommendation: str) -> float:
        """Simplified expected value calculation."""
        action_values = {
            "raise": 1.0,
            "bet": 0.8,
            "check/call": 0.3,
            "check/fold": 0.0,
            "call": 0.5,
            "fold": 0.0,
        }
        return win_prob * action_values.get(recommendation, 0.0)


# Smoke: same spots under random vs strong vs value
if __name__ == "__main__":
    engine = PokerEngine(n_simulations=800)
    spots = [
        (["As", "4h"], ["4c", "9d", "2s"], "A4 on 4-high flop"),
        (["As", "Ah"], ["Kd", "7c", "2h"], "AA on dry K72"),
        (["As", "Ks"], ["Tc", "Jd", "2s"], "AKs on TcJd2s"),
    ]
    modes = ("random", "strong", "value")
    for holes, board, label in spots:
        print(f"\n{label}: {holes} / {board}")
        for mode in modes:
            r = engine.evaluate_hand(holes, board, villain_range=mode)
            print(f"  {mode:7s}  win={r.win_probability:6.1%}  {r.recommendation}")
