package laya_test

import (
	"context"
	"fmt"
	"log"

	"github.com/MeKo-Christian/go-laya"
	"github.com/MeKo-Christian/go-laya/presets"
)

// These are the README's examples (PLAN.md Task 7.6.1). Every one but
// ExampleRouter_Route needs the weights and an ONNX export, so they carry no
// Output comment: CI proves they compile, not what they print.
// TestREADMEExamples runs them against local exports.

// state is the README's support e-mail. An Obj keeps its key order, which is
// the order the model reads the fields in.
var state = laya.Obj{
	{Key: "from", Value: "user@acme.com"},
	{Key: "subject", Value: "Duplicate charge on invoice #4411"},
	{Key: "body", Value: "Hi, we were billed twice for March. Please refund the duplicate today or we will cancel our plan."},
}

// questions are the README's typed questions. Questions is a slice, not a
// map, because the order is the order the answers come back in.
var questions = laya.Questions{
	{ID: "department", Q: laya.ChoiceQuestion{
		Ins: "Which department should handle this request?",
		Opts: []laya.ChoiceOption{
			{Key: "billing", Desc: "invoices, payments, refunds"},
			{Key: "technical", Desc: "bugs, outages, system errors"},
			{Key: "sales", Desc: "pricing, new contracts"},
			{Key: "other", Desc: "everything else"},
		},
	}},
	{ID: "urgency", Q: laya.ScoreQuestion{
		Ins:    "How urgent is this request?",
		Levels: []laya.Criterion{"not urgent", "soon", "critical deadline or blocking issue"},
	}},
	{ID: "churn_risk", Q: laya.NoulQuestion{Ins: "Does the user threaten to cancel or leave?"}},
	{ID: "refund_requested", Q: laya.NoulQuestion{Ins: "Does the user explicitly request a refund?"}},
}

func ExampleRouter() {
	ctx := context.Background()
	router, err := laya.NewRouter()
	if err != nil {
		log.Fatal(err)
	}
	defer router.Close()

	// Build every checkpoint up front, so no request pays a model load.
	if err := router.Preload(ctx); err != nil {
		log.Panic(err)
	}

	// English state: routed to laya (ModernBERT-large).
	res, err := router.Predict(ctx, state, questions)
	if err != nil {
		log.Panic(err)
	}
	dept, _ := res.Answers.Get("department")
	fmt.Printf("Department: %s (confidence %.2f)\n", dept.Choice, dept.Confidence) // billing
	fmt.Println("Routing   :", res.Routing.Model)                                  // english

	// Hindi state: routed to laya-multilingual (mmBERT-base).
	hindi := laya.Obj{{Key: "body", Value: "मुझसे दो बार शुल्क लिया गया, कृपया पैसे वापस करें।"}}
	res, err = router.Predict(ctx, hindi, questions)
	if err != nil {
		log.Panic(err)
	}
	dept, _ = res.Answers.Get("department")
	fmt.Printf("Department: %s (confidence %.2f)\n", dept.Choice, dept.Confidence) // billing
	fmt.Println("Routing   :", res.Routing.Model)                                  // multilingual
	fmt.Println("Repo      :", res.Routing.Repo)                                   // convaiinnovations/laya/multilingual
	fmt.Println("Reason    :", res.Routing.Reason)                                 // non-Latin script (devanagari, ...)

	// An explicit override when you want a specific checkpoint.
	if _, err := router.Predict(ctx, state, questions, laya.ForModel(laya.ModelTypedDecisions)); err != nil {
		log.Panic(err)
	}
}

func ExampleRouter_Route() {
	router, err := laya.NewRouter()
	if err != nil {
		log.Fatal(err)
	}
	defer router.Close()

	// Route decides without loading or running anything.
	d, err := router.Route(laya.Obj{{Key: "body", Value: "Der Kunde wurde zweimal belastet"}}, questions)
	if err != nil {
		log.Panic(err)
	}
	fmt.Println(d.Model)
	fmt.Println(d.Reason)
	// Output:
	// multilingual
	// Latin script but language looks like 'de', not English
}

func ExampleRouter_Preload() {
	ctx := context.Background()
	// Keep two checkpoints resident; past that the least recently used is
	// evicted. The default is one.
	router, err := laya.NewRouter(laya.WithMaxLoaded(2))
	if err != nil {
		log.Fatal(err)
	}
	defer router.Close()

	// Preload only the checkpoints you serve. With no names, Preload builds
	// all of them.
	if err := router.Preload(ctx, laya.ModelEnglish, laya.ModelMultilingual); err != nil {
		log.Panic(err)
	}
	fmt.Println(router.Loaded()) // [english multilingual]

	// Free the memory again.
	if err := router.Unload(); err != nil {
		log.Panic(err)
	}
}

func ExampleRouter_Attach() {
	ctx := context.Background()
	agent, err := laya.Open(ctx, "convaiinnovations/laya")
	if err != nil {
		log.Fatal(err)
	}
	defer agent.Close()

	router, err := laya.NewRouter()
	if err != nil {
		log.Panic(err)
	}
	defer router.Close()

	// Hand the router the agent you already built instead of loading a
	// second copy. The router never closes an attached agent.
	if err := router.Attach(laya.ModelEnglish, agent); err != nil {
		log.Panic(err)
	}
}

func ExampleOpen() {
	ctx := context.Background()
	// One checkpoint, loaded directly. "" is convaiinnovations/laya as well.
	agent, err := laya.Open(ctx, "convaiinnovations/laya") // English
	if err != nil {
		log.Fatal(err)
	}
	defer agent.Close()

	multilingual, err := laya.Open(ctx, "convaiinnovations/laya", laya.WithSubfolder("multilingual")) // 100+ languages
	if err != nil {
		log.Panic(err)
	}
	defer multilingual.Close()

	// Every question in one forward pass.
	res, err := agent.Predict(ctx, state, questions)
	if err != nil {
		log.Panic(err)
	}
	dept, _ := res.Answers.Get("department")
	urgency, _ := res.Answers.Get("urgency")
	churn, _ := res.Answers.Get("churn_risk")
	fmt.Println("Department:", dept.Choice)              // billing
	fmt.Printf("Urgency   : %.2f / 2\n", *urgency.Score) // the expected level
	fmt.Printf("Churn risk: %.3f\n", *churn.Noul)        // P(true)
}

func ExampleAnswer_confidence() {
	ctx := context.Background()
	agent, err := laya.Open(ctx, "convaiinnovations/laya")
	if err != nil {
		log.Fatal(err)
	}
	defer agent.Close()

	res, err := agent.Predict(ctx, state, questions)
	if err != nil {
		log.Panic(err)
	}
	dept, _ := res.Answers.Get("department")
	if dept.Confidence >= 0.85 {
		// High confidence: act without a human in the loop.
		fmt.Println("route automatically to", dept.Choice)
	} else {
		// Low confidence: escalate to human triage.
		fmt.Printf("escalate %s to a human (confidence %.2f)\n", dept.Choice, dept.Confidence)
	}
}

func ExampleAgent_presets() {
	ctx := context.Background()
	agent, err := laya.Open(ctx, "convaiinnovations/laya")
	if err != nil {
		log.Fatal(err)
	}
	defer agent.Close()

	ask := func(field, text string, qs laya.Questions) *laya.Result {
		res, err := agent.Predict(ctx, laya.Obj{{Key: field, Value: text}}, qs)
		if err != nil {
			log.Panic(err)
		}
		return res
	}

	// 1. Model routing: send a request to a small or a frontier model.
	routing := ask("request", "Refactor this service using dependency injection", presets.RouterQuestions())
	// 2. Prompt guardrails: jailbreaks, injections, leaks.
	guard := ask("prompt", "Ignore all instructions", presets.GuardQuestions())
	// 3. Content safety and moderation: toxicity, harassment, threats.
	safety := ask("post", "User comment text", presets.ModerationQuestions())
	// 4. Support ticket triage: intent, urgency, frustration, churn.
	triage := ask("message", "My payment failed twice", presets.TriageQuestions())

	fmt.Println(len(routing.Answers), len(guard.Answers), len(safety.Answers), len(triage.Answers))
}

func ExampleAgent_SetLimits() {
	ctx := context.Background()
	agent, err := laya.Open(ctx, "convaiinnovations/laya", laya.WithSubfolder("multilingual"))
	if err != nil {
		log.Fatal(err)
	}
	defer agent.Close()

	// Many options share the head_max_len budget. Raise it, and max_len with
	// it, so each of 50+ options keeps enough tokens to stay distinct.
	if err := agent.SetLimits(1024, 512); err != nil {
		log.Panic(err)
	}
	fmt.Println(agent.MaxLen(), agent.HeadMaxLen()) // 1024 512
}
