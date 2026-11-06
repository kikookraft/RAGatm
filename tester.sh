#!/usr/bin/env bash
# Manual test battery for the "RAG against the machine" project.
# Section A: crash tests — degenerate inputs, must fail CLEANLY
#            (clear message + exit code != 0), never a raw stack trace.
# Section B: functional tests — full pipeline, must produce correct results.
# Section C: benchmarks — must respect the time limits from the subject (VI.1.2).
#
# Usage:
#   ./test_battery.sh a   # crash tests only
#   ./test_battery.sh b   # functional tests only
#   ./test_battery.sh c   # benchmarks only
#   ./test_battery.sh d   # evaluate only (moulinette + your own)
#   ./test_battery.sh     # all

set -u
RUN=(uv run python3 -m src)
TMP_DIR="/tmp/rag_test_battery"
mkdir -p "$TMP_DIR"

DOCS_UNANSWERED="data/datasets/UnansweredQuestions/dataset_docs_public.json"
DOCS_ANSWERED="data/datasets/AnsweredQuestions/dataset_docs_public.json"
CODE_UNANSWERED="data/datasets/UnansweredQuestions/dataset_code_public.json"
CODE_ANSWERED="data/datasets/AnsweredQuestions/dataset_code_public.json"

PASS_COUNT=0
FAIL_COUNT=0

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[1;36m'
MAGENTA='\033[0;35m'
NC='\033[0m'

section() {
    echo ""
    echo -e "${CYAN}################################################################${NC}"
    echo -e "${CYAN}# $1${NC}"
    echo -e "${CYAN}################################################################${NC}"
}

# Prints what THE TESTER is doing/expecting (yellow), then opens a clearly
# delimited zone (magenta) for the ACTUAL OUTPUT of the program.
test_case() {
    echo ""
    echo -e "${YELLOW}>>> TEST: $1${NC}"
    if [ -n "${2:-}" ]; then
        echo -e "${YELLOW}--- expected: $2${NC}"
    fi
    echo -e "${MAGENTA}┌─ program output ──────────────────────────────────────────${NC}"
}

end_output() {
    echo -e "${MAGENTA}└────────────────────────────────────────────────────────────${NC}"
}

# Runs a command, closes the output box, THEN prints the elapsed time
# separately (so it's never mixed in with the program's own output).
timed_run() {
    local start end elapsed
    start=$(date +%s.%N)
    "$@"
    end=$(date +%s.%N)
    end_output
    elapsed=$(awk -v s="$start" -v e="$end" 'BEGIN { printf "%.3f", e - s }')
    echo -e "${CYAN}Elapsed time: ${elapsed}s${NC}"
}

# Same as timed_run, but also validates the elapsed time against a threshold
# (in seconds) and prints an OK/KO message, updating PASS_COUNT/FAIL_COUNT.
timed_run_check() {
    local threshold="$1"
    shift
    local start end elapsed
    start=$(date +%s.%N)
    "$@"
    end=$(date +%s.%N)
    end_output
    elapsed=$(awk -v s="$start" -v e="$end" 'BEGIN { printf "%.3f", e - s }')
    echo -e "${CYAN}Elapsed time: ${elapsed}s (threshold: ${threshold}s)${NC}"
    if awk -v e="$elapsed" -v t="$threshold" 'BEGIN { exit !(e <= t) }'; then
        echo -e "${GREEN}✅ OK — within threshold${NC}"
        PASS_COUNT=$((PASS_COUNT + 1))
    else
        echo -e "${RED}❌ KO — exceeds threshold${NC}"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
}

# Verifies that the project structure is present before running any test.
# Exits early with a clear message instead of letting every single test
# fail confusingly because uv/src/the dataset files are missing.
# Usage: check_prerequisites [need_moulinette]
check_prerequisites() {
    local need_moulinette="${1:-false}"
    local missing=0

    if ! command -v uv >/dev/null 2>&1; then
        echo -e "${RED}❌ 'uv' command not found — is it installed and in your PATH?${NC}"
        missing=1
    fi

    if [ ! -f "pyproject.toml" ]; then
        echo -e "${RED}❌ 'pyproject.toml' not found — run this script from the project root.${NC}"
        missing=1
    fi

    if [ ! -d "src" ]; then
        echo -e "${RED}❌ 'src/' directory not found — run this script from the project root.${NC}"
        missing=1
    fi

    for f in "$DOCS_UNANSWERED" "$DOCS_ANSWERED" "$CODE_UNANSWERED" "$CODE_ANSWERED"; do
        if [ ! -f "$f" ]; then
            echo -e "${RED}❌ Dataset file not found: $f${NC}"
            missing=1
        fi
    done

    if [ "$need_moulinette" = "true" ] && [ ! -x "./moulinette" ]; then
        echo -e "${RED}❌ './moulinette' not found or not executable.${NC}"
        missing=1
    fi

    if [ "$missing" -ne 0 ]; then
        echo -e "${RED}Aborting: missing prerequisites, see above.${NC}"
        exit 1
    fi
}

# Runs a command, captures its output, and validates automatically:
# KO only if a real Python traceback appears in the output.
# Otherwise (handled error or success), it's considered valid.
check_no_crash() {
    local description="$1"
    shift
    local output
    output=$("$@" 2>&1)
    if echo "$output" | grep -q "Traceback (most recent call last)"; then
        echo -e "${RED}❌ KO — $description${NC}"
        echo "$output" | tail -25
        FAIL_COUNT=$((FAIL_COUNT + 1))
    else
        echo -e "${GREEN}✅ OK — $description${NC}"
        PASS_COUNT=$((PASS_COUNT + 1))
    fi
}

run_crash_tests() {
    check_prerequisites

    section "SECTION A — CRASH TESTS (automatic validation: KO only if Traceback)"

    PASS_COUNT=0
    FAIL_COUNT=0

    # --- Fixtures for malformed file tests ---
    echo "not valid json at all {{{" > "$TMP_DIR/malformed.json"
    echo '{"wrong_key": "wrong_schema"}' > "$TMP_DIR/wrong_schema.json"
    echo '{"rag_questions": []}' > "$TMP_DIR/empty_dataset.json"
    mkdir -p "$TMP_DIR/a_directory"

    # ===== index =====
    check_no_crash "index --max_chunk_size 0" "${RUN[@]}" index --max_chunk_size 0
    check_no_crash "index --max_chunk_size -5" "${RUN[@]}" index --max_chunk_size -5
    check_no_crash "index --max_chunk_size abc" "${RUN[@]}" index --max_chunk_size abc
    check_no_crash "index --max_chunk_size 1.5" "${RUN[@]}" index --max_chunk_size 1.5
    check_no_crash "index --max_chunk_size (flag with no value)" "${RUN[@]}" index --max_chunk_size

    # ===== search =====
    check_no_crash "search (no argument)" "${RUN[@]}" search
    check_no_crash 'search ""' "${RUN[@]}" search ""
    check_no_crash 'search "   "' "${RUN[@]}" search "   "
    check_no_crash "search --query (flag with no value) --k 5" "${RUN[@]}" search --query --k 5
    check_no_crash 'search "test" --k 0' "${RUN[@]}" search "test" --k 0
    check_no_crash 'search "test" --k -1' "${RUN[@]}" search "test" --k -1
    check_no_crash 'search "test" --k abc' "${RUN[@]}" search "test" --k abc
    check_no_crash 'search "test" --k 1.5' "${RUN[@]}" search "test" --k 1.5
    check_no_crash 'search "test" --k 999999' "${RUN[@]}" search "test" --k 999999
    check_no_crash 'search unicode/emoji' "${RUN[@]}" search "🚀🚀🚀 中文 emoji test"
    check_no_crash 'search query 5000 chars' "${RUN[@]}" search "$(python3 -c 'print("a"*5000)')"

    # ===== search_dataset =====
    check_no_crash "search_dataset nonexistent file" \
        "${RUN[@]}" search_dataset /nonexistent/path.json --save_directory "$TMP_DIR/out"
    check_no_crash "search_dataset directory instead of file" \
        "${RUN[@]}" search_dataset "$TMP_DIR/a_directory" --save_directory "$TMP_DIR/out"
    check_no_crash "search_dataset malformed JSON" \
        "${RUN[@]}" search_dataset "$TMP_DIR/malformed.json" --save_directory "$TMP_DIR/out"
    check_no_crash "search_dataset wrong schema" \
        "${RUN[@]}" search_dataset "$TMP_DIR/wrong_schema.json" --save_directory "$TMP_DIR/out"
    check_no_crash "search_dataset empty dataset" \
        "${RUN[@]}" search_dataset "$TMP_DIR/empty_dataset.json" --save_directory "$TMP_DIR/out"
    check_no_crash "search_dataset --k 0" \
        "${RUN[@]}" search_dataset "$DOCS_UNANSWERED" --k 0 --save_directory "$TMP_DIR/out"

    # ===== answer =====
    check_no_crash 'answer ""' "${RUN[@]}" answer ""
    check_no_crash 'answer off-topic question' "${RUN[@]}" answer "How to bake cookies?"
    check_no_crash 'answer "test" --k 0' "${RUN[@]}" answer "test" --k 0

    # ===== answer_dataset =====
    check_no_crash "answer_dataset nonexistent file" \
        "${RUN[@]}" answer_dataset /nonexistent/path.json --save_directory "$TMP_DIR/out_answers"
    check_no_crash "answer_dataset malformed JSON" \
        "${RUN[@]}" answer_dataset "$TMP_DIR/malformed.json" --save_directory "$TMP_DIR/out_answers"
    check_no_crash "answer_dataset wrong schema" \
        "${RUN[@]}" answer_dataset "$TMP_DIR/wrong_schema.json" --save_directory "$TMP_DIR/out_answers"

    # ===== evaluate =====
    check_no_crash "evaluate nonexistent files" \
        "${RUN[@]}" evaluate /nonexistent.json /nonexistent2.json
    check_no_crash "evaluate malformed JSON on student side" \
        "${RUN[@]}" evaluate "$TMP_DIR/malformed.json" "$DOCS_ANSWERED"

    "${RUN[@]}" search_dataset "$CODE_UNANSWERED" --save_directory "$TMP_DIR/out" >/dev/null 2>&1
    check_no_crash "evaluate (0 question_id in common, code vs docs)" \
        "${RUN[@]}" evaluate "$TMP_DIR/out/dataset_code_public.json" "$DOCS_ANSWERED"

    check_no_crash "evaluate --k 0" \
        "${RUN[@]}" evaluate "$TMP_DIR/out/dataset_code_public.json" "$CODE_ANSWERED" --k 0
    check_no_crash "evaluate --max_context_length 0" \
        "${RUN[@]}" evaluate "$TMP_DIR/out/dataset_code_public.json" "$CODE_ANSWERED" --max_context_length 0
    check_no_crash "evaluate --max_context_length -100" \
        "${RUN[@]}" evaluate "$TMP_DIR/out/dataset_code_public.json" "$CODE_ANSWERED" --max_context_length -100

    section "END SECTION A — Summary: ${GREEN}$PASS_COUNT OK${NC} / ${RED}$FAIL_COUNT KO${NC}"
}

run_functional_tests() {
    check_prerequisites true

    section "SECTION B — FUNCTIONAL TESTS (full pipeline, must succeed)"

    test_case "index --max_chunk_size 2000" "must finish without error, < 5 min"
    timed_run "${RUN[@]}" index --max_chunk_size 2000

    test_case "search docs question" "must return coherent sources"
    "${RUN[@]}" search "How to configure OpenAI server?" --k 5
    end_output

    test_case "search_dataset (docs)" "must process all 100 questions"
    "${RUN[@]}" search_dataset "$DOCS_UNANSWERED" --k 10 --save_directory data/output/search_results
    end_output

    test_case "search_dataset (code)" "must process all 100 questions"
    "${RUN[@]}" search_dataset "$CODE_UNANSWERED" --k 10 --save_directory data/output/search_results
    end_output

    test_case "evaluate (docs) — expected threshold: recall@5 >= 80%" "moulinette then your own command, the numbers must match"
    ./moulinette evaluate_student_search_results \
        --student_answer_path data/output/search_results/dataset_docs_public.json \
        --dataset_path "$DOCS_ANSWERED" --k 10 --max_context_length 2000
    echo -e "${MAGENTA}--- your own evaluate command ---${NC}"
    "${RUN[@]}" evaluate data/output/search_results/dataset_docs_public.json "$DOCS_ANSWERED"
    end_output

    test_case "evaluate (code) — expected threshold: recall@5 >= 50%" "moulinette then your own command, the numbers must match"
    ./moulinette evaluate_student_search_results \
        --student_answer_path data/output/search_results/dataset_code_public.json \
        --dataset_path "$CODE_ANSWERED" --k 10 --max_context_length 2000
    echo -e "${MAGENTA}--- your own evaluate command ---${NC}"
    "${RUN[@]}" evaluate data/output/search_results/dataset_code_public.json "$CODE_ANSWERED"

    test_case "answer single query" "coherent, short answer, based on context"
    "${RUN[@]}" answer "How to configure OpenAI server?" --k 10
    end_output

    test_case "answer_dataset (docs sample)" "must produce a valid JSON with answer filled in"
    "${RUN[@]}" answer_dataset data/output/search_results/dataset_docs_public.json \
        --save_directory data/output/search_results_and_answer
    end_output

    section "END SECTION B"
}

run_benchmarks() {
    check_prerequisites

    section "SECTION C — BENCHMARKS (limits from the subject VI.1.2)"

    PASS_COUNT=0
    FAIL_COUNT=0

    test_case "Indexing time" "must be < 5 minutes (300s)"
    timed_run_check 300 "${RUN[@]}" index --max_chunk_size 2000

    test_case "Cold start (first answer, loads the model)" "must be < 60 seconds"
    timed_run_check 60 "${RUN[@]}" answer "How to configure OpenAI server?" --k 5

    test_case "Warm retrieval throughput (search_dataset, 100 docs questions)" \
        "must be well under 90s for 1000 questions (so < 9s for 100, extrapolating linearly)"
    timed_run_check 9 "${RUN[@]}" search_dataset "$DOCS_UNANSWERED" --k 10 --save_directory "$TMP_DIR/out"

    section "END SECTION C — Summary: ${GREEN}$PASS_COUNT OK${NC} / ${RED}$FAIL_COUNT KO${NC}"
}

run_evaluate_tests() {
    check_prerequisites true

    section "SECTION D — EVALUATE (moulinette + your own evaluate, docs & code)"

    PASS_COUNT=0
    FAIL_COUNT=0

    test_case "index --max_chunk_size 2000" "must finish without error before searching"
    timed_run "${RUN[@]}" index --max_chunk_size 2000

    test_case "search_dataset (docs)" "generating search results for evaluation"
    "${RUN[@]}" search_dataset "$DOCS_UNANSWERED" --k 10 --save_directory data/output/search_results
    end_output

    test_case "search_dataset (code)" "generating search results for evaluation"
    "${RUN[@]}" search_dataset "$CODE_UNANSWERED" --k 10 --save_directory data/output/search_results
    end_output

    test_case "evaluate (docs) — expected threshold: recall@5 >= 80%" "moulinette then your own command, the numbers must match"
    ./moulinette evaluate_student_search_results \
        --student_answer_path data/output/search_results/dataset_docs_public.json \
        --dataset_path "$DOCS_ANSWERED" --k 10 --max_context_length 2000
    echo -e "${MAGENTA}--- your own evaluate command ---${NC}"
    "${RUN[@]}" evaluate data/output/search_results/dataset_docs_public.json "$DOCS_ANSWERED"
    end_output

    test_case "evaluate (code) — expected threshold: recall@5 >= 50%" "moulinette then your own command, the numbers must match"
    ./moulinette evaluate_student_search_results \
        --student_answer_path data/output/search_results/dataset_code_public.json \
        --dataset_path "$CODE_ANSWERED" --k 10 --max_context_length 2000
    echo -e "${MAGENTA}--- your own evaluate command ---${NC}"
    "${RUN[@]}" evaluate data/output/search_results/dataset_code_public.json "$CODE_ANSWERED" 
    end_output

    section "END SECTION D"
}

case "${1:-all}" in
    a) run_crash_tests ;;
    b) run_functional_tests ;;
    c) run_benchmarks ;;
    d) run_evaluate_tests ;;
    all)
        run_crash_tests
        run_functional_tests
        run_benchmarks
        run_evaluate_tests
        ;;
    *)
        echo "Usage: $0 [a|b|c|d|all]"
        exit 1
        ;;
esac