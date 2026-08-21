"""The terminal prompts: the pure menu loop, the key decoder, the numbered fallback.

No test here opens a terminal. :func:`~app.cli.prompts.run_menu` takes decoded key names
and a draw callback, :func:`~app.cli.prompts.decode_keys` takes a read function, and
:class:`~app.cli.prompts.Terminal` takes its two streams — which is the whole reason the
picker is split that way.
"""

from __future__ import annotations

import io
from collections.abc import Iterable

from app.cli.prompts import (
    ABORT,
    CURSOR,
    DOWN,
    ENTER,
    UP,
    Choice,
    Terminal,
    decode_keys,
    run_menu,
)

ROWS = ("first", "second", "third")


def choices() -> list[Choice[str]]:
    return [Choice(value=row, label=row.title(), detail=f"{len(row)} letters") for row in ROWS]


def menu(keys: Iterable[str], count: int = 3) -> tuple[int | None, list[int]]:
    """Run the loop, returning what it chose and every cursor position it drew."""
    drawn: list[int] = []
    chosen = run_menu(count, keys, draw=drawn.append)
    return chosen, drawn


# --- the loop ---------------------------------------------------------------


def test_the_first_row_is_drawn_before_a_key_is_pressed() -> None:
    assert menu([ENTER]) == (0, [0])


def test_down_then_enter_takes_the_second_row() -> None:
    assert menu([DOWN, ENTER]) == (1, [0, 1])


def test_the_cursor_wraps_so_abort_is_one_key_away() -> None:
    """Abort is the last row, which makes ↑ from the top the fastest way out."""
    assert menu([UP, ENTER]) == (2, [0, 2])
    assert menu([DOWN, DOWN, DOWN, ENTER]) == (0, [0, 1, 2, 0])


def test_a_digit_jumps_straight_to_a_row() -> None:
    assert menu(["#3", ENTER]) == (2, [0, 2])


def test_a_digit_past_the_end_is_ignored_rather_than_clamped() -> None:
    """Clamping would choose a film the user did not ask for."""
    assert menu(["#9", ENTER]) == (0, [0])


def test_aborting_chooses_nothing() -> None:
    assert menu([DOWN, ABORT]) == (None, [0, 1])


def test_input_running_out_is_not_a_choice() -> None:
    assert menu([DOWN]) == (None, [0, 1])


def test_unknown_keys_do_not_move_the_cursor() -> None:
    assert menu(["other", "\x07", ENTER]) == (0, [0])


def test_an_empty_menu_is_never_drawn() -> None:
    assert menu([ENTER], count=0) == (None, [])


# --- the key decoder --------------------------------------------------------


def decoded(raw: bytes, *, waiting: bool = True) -> list[str]:
    stream = io.BytesIO(raw)
    return list(decode_keys(stream.read, waiting=lambda: waiting))


def test_arrows_move_in_both_of_the_two_spellings() -> None:
    assert decoded(b"\x1b[A\x1b[B\x1bOA\x1bOB") == [UP, DOWN, UP, DOWN]


def test_vi_keys_move_too() -> None:
    assert decoded(b"kjKJ") == [UP, DOWN, UP, DOWN]


def test_enter_accepts_either_line_ending() -> None:
    assert decoded(b"\r\n") == [ENTER, ENTER]


def test_q_and_the_two_control_keys_abort() -> None:
    assert decoded(b"qQ\x03\x04") == [ABORT, ABORT, ABORT, ABORT]


def test_digits_become_jumps_but_zero_does_not() -> None:
    assert decoded(b"19") == ["#1", "#9"]
    assert decoded(b"0") == ["other"]


def test_a_lone_escape_aborts_and_an_escape_sequence_does_not() -> None:
    """The two start with the same byte; only what follows tells them apart."""
    assert decoded(b"\x1b", waiting=False) == [ABORT]
    assert decoded(b"\x1b[A", waiting=True) == [UP]


def test_an_unrecognised_escape_sequence_is_just_ignored() -> None:
    assert decoded(b"\x1b[5~") == ["other", "other"]


# --- the numbered fallback --------------------------------------------------


def paper_terminal(typed: str = "") -> tuple[Terminal, io.StringIO]:
    """A terminal on two string buffers: no TTY, so the numbered prompt is used."""
    err = io.StringIO()
    return Terminal(stdin=io.StringIO(typed), stderr=err), err


def test_without_a_tty_the_choices_are_numbered() -> None:
    terminal, err = paper_terminal("2\n")

    chosen = terminal.select("Which film is this?", choices())

    assert chosen is not None and chosen.value == "second"
    assert not terminal.interactive
    printed = err.getvalue()
    assert "Which film is this?" in printed
    assert "1) First  5 letters" in printed
    assert CURSOR not in printed  # nothing to move, so nothing pretends there is


def test_an_empty_answer_takes_the_first_choice() -> None:
    terminal, _ = paper_terminal("\n")

    chosen = terminal.select("Which film is this?", choices())

    assert chosen is not None and chosen.value == "first"


def test_three_bad_answers_end_the_prompt_rather_than_looping() -> None:
    """A pipe that keeps saying "no" must not spin forever."""
    terminal, err = paper_terminal("nine\n0\n47\n2\n")

    assert terminal.select("Which film is this?", choices()) is None
    assert err.getvalue().count("is not one of them") == 3


def test_end_of_input_takes_the_first_choice_and_stops_asking() -> None:
    terminal, _ = paper_terminal("")

    chosen = terminal.select("Which film is this?", choices())

    assert chosen is not None and chosen.value == "first"


def test_an_empty_list_is_not_a_question() -> None:
    terminal, err = paper_terminal("1\n")

    assert terminal.select("Which film is this?", []) is None
    assert err.getvalue() == ""


def test_a_typed_line_comes_back_stripped() -> None:
    terminal, err = paper_terminal("  Blade Runner  \n")

    assert terminal.ask("Title:") == "Blade Runner"
    assert err.getvalue() == "Title: "


def test_a_paste_is_everything_up_to_the_end_of_input() -> None:
    terminal, err = paper_terminal("Negative format\n35 mm\n")

    assert terminal.paste("Paste, then Ctrl-D.") == "Negative format\n35 mm\n"
    assert "Paste, then Ctrl-D." in err.getvalue()
