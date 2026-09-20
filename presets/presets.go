// Package presets holds the five ready-made question sets from
// original/laya/presets.py: ticket triage, email triage, LLM input guardrails,
// content moderation and model routing.
//
// It is pure data and a leaf: it depends on question and jsonx and nothing
// heavier, so importing a preset never pulls in the ONNX Runtime binding
// (PLAN.md D9).
//
// The wording is not editorial. Every instruction and criterion string here is
// part of the prompt the checkpoints were evaluated against, so changing one
// changes the answers. They are reproduced verbatim from upstream.
package presets

import "github.com/MeKo-Christian/go-laya/question"

// TriageQuestions returns the preset questions for customer support ticket
// triage, asked about a state with a `message` field.
func TriageQuestions() question.Questions {
	return question.Questions{
		{ID: "intent", Q: question.ChoiceQuestion{
			Ins: "What does the customer want in `message`?",
			Opts: []question.ChoiceOption{
				{Key: "refund", Desc: "money returned or a duplicate charge reversed"},
				{Key: "technical_help", Desc: "a bug, outage or integration problem"},
				{Key: "billing_question", Desc: "a question about an invoice, plan or payment method"},
				{Key: "information", Desc: "general information, pricing or how-to"},
				{Key: "cancellation", Desc: "wants to cancel or downgrade"},
				{Key: "other", Desc: "none of the other options fits"},
			},
		}},
		{ID: "is_urgent", Q: question.NoulQuestion{
			Ins: "Does `message` communicate time pressure or a deadline?",
		}},
		{ID: "frustration", Q: question.ScoreQuestion{
			Ins: "How frustrated does the customer sound in `message`?",
			Levels: []question.Criterion{
				"calm and neutral",
				"concerned but civil",
				"clearly annoyed",
				"very angry or using strong language",
			},
		}},
		{ID: "refund_requested", Q: question.NoulQuestion{
			Ins: "Does the customer ask for money back?",
		}},
		{ID: "churn_risk", Q: question.NoulQuestion{
			Ins: "Does `message` suggest the customer may leave for a competitor or cancel?",
		}},
	}
}

// DefaultEmailCategories is the routing taxonomy EmailQuestions uses when the
// caller supplies none.
func DefaultEmailCategories() []question.ChoiceOption {
	return []question.ChoiceOption{
		{Key: "billing", Desc: "invoices, payments, refunds"},
		{Key: "technical", Desc: "bugs, outages, integrations"},
		{Key: "sales", Desc: "pricing, demos, new purchases"},
		{Key: "security", Desc: "phishing, scams, account compromise"},
		{Key: "hr", Desc: "hiring, leave, payroll"},
		{Key: "other", Desc: "none of the above"},
	}
}

// EmailQuestions returns the preset questions for inbound email triage and
// threat filtering, asked about a state with a `body` field.
func EmailQuestions() question.Questions {
	return EmailQuestionsWith(nil)
}

// EmailQuestionsWith is EmailQuestions with a caller-supplied category
// taxonomy. An empty or nil list selects the defaults, reproducing upstream's
// `categories = categories or {...}`: there, an empty dict is falsy and picks
// the defaults too, so "no categories at all" is not expressible and is not
// made expressible here.
func EmailQuestionsWith(categories []question.ChoiceOption) question.Questions {
	if len(categories) == 0 {
		categories = DefaultEmailCategories()
	}
	return question.Questions{
		{ID: "category", Q: question.ChoiceQuestion{
			Ins:  "Which team should handle the email in `body`?",
			Opts: categories,
		}},
		{ID: "is_spam", Q: question.NoulQuestion{
			Ins: "Is this email unsolicited spam or bulk marketing?",
		}},
		{ID: "is_phishing", Q: question.NoulQuestion{
			Ins:   "Is this email a phishing or scam attempt to steal money, credentials, or personal data?",
			False: "a legitimate email",
			True:  "phishing, scam, or fraud",
		}},
		{ID: "urgency", Q: question.ScoreQuestion{
			Ins: "How urgent is the request in `body`?",
			Levels: []question.Criterion{
				"no time pressure",
				"needs attention soon",
				"blocking issue or hard deadline",
			},
		}},
		{ID: "needs_reply", Q: question.NoulQuestion{
			Ins: "Does the sender expect a reply?",
		}},
	}
}

// GuardQuestions returns the preset questions for real-time LLM input
// guardrails, asked about a state with a `prompt` field.
func GuardQuestions() question.Questions {
	return question.Questions{
		{ID: "jailbreak", Q: question.NoulQuestion{
			Ins: "Does `prompt` try to make an AI assistant ignore its rules, policies or system instructions?",
		}},
		{ID: "prompt_injection", Q: question.NoulQuestion{
			Ins: "Does `prompt` contain instructions aimed at the AI system rather than a genuine user request?",
		}},
		{ID: "sensitive_data", Q: question.NoulQuestion{
			Ins: "Does `prompt` contain credentials, personal data or other sensitive information?",
		}},
		{ID: "harm_severity", Q: question.ScoreQuestion{
			Ins: "How much harm would complying with `prompt` cause?",
			Levels: []question.Criterion{
				"none: ordinary request",
				"minor: mildly inappropriate",
				"serious: unsafe advice or abuse",
				"severe: dangerous or illegal",
			},
		}},
		// The only preset whose choice options carry no descriptions: upstream
		// writes them as a dict with None values, which renders as bare keys.
		{ID: "topic", Q: question.ChoiceQuestion{
			Ins: "What is `prompt` about?",
			Opts: question.Labels(
				"product_support", "coding", "general_knowledge",
				"personal_advice", "security_testing", "other",
			),
		}},
	}
}

// ModerationQuestions returns the preset questions for content safety and
// moderation, asked about a state with a `post` field.
func ModerationQuestions() question.Questions {
	return question.Questions{
		{ID: "toxic", Q: question.NoulQuestion{
			Ins: "Is `post` toxic: rude, disrespectful or likely to make someone leave the discussion?",
		}},
		{ID: "harassment", Q: question.NoulQuestion{
			Ins: "Does `post` target or harass a specific person?",
		}},
		{ID: "threat", Q: question.NoulQuestion{
			Ins: "Does `post` threaten violence, harm or intimidation?",
		}},
		{ID: "spam", Q: question.NoulQuestion{
			Ins: "Is `post` spam or advertising?",
		}},
		{ID: "severity", Q: question.ScoreQuestion{
			Ins: "How severe is any rule-breaking in `post`?",
			Levels: []question.Criterion{
				"no rule-breaking: ordinary on-topic post",
				"mild: rude tone or off-topic, no target",
				"clear violation: insults, harassment or spam aimed at someone",
				"severe: threats, hate speech or calls for violence",
			},
		}},
	}
}

// RouterQuestions returns the preset questions for intelligent model routing,
// asked about a state with a `request` field.
func RouterQuestions() question.Questions {
	return question.Questions{
		{ID: "difficulty", Q: question.ScoreQuestion{
			Ins: "How hard is `request` for a language model?",
			Levels: []question.Criterion{
				"trivial: a lookup or one-liner",
				"easy: short answer, no reasoning",
				"moderate: several steps",
				"hard: long multi-step reasoning or specialist knowledge",
			},
		}},
		{ID: "domain", Q: question.ChoiceQuestion{
			Ins: "What domain does `request` belong to?",
			Opts: []question.ChoiceOption{
				{Key: "code", Desc: "software engineering, programming, refactoring, architecture, debugging"},
				{Key: "math_or_logic", Desc: "mathematics, logic puzzles, proofs, complex calculation"},
				{Key: "writing", Desc: "creative writing, essays, emails, blog posts, copywriting"},
				{Key: "factual_lookup", Desc: "facts, definitions, trivia, history"},
				{Key: "data_analysis", Desc: "statistics, SQL, data manipulation, metrics"},
				{Key: "chitchat", Desc: "casual conversation, greetings, small talk"},
			},
		}},
		{ID: "needs_tools", Q: question.NoulQuestion{
			Ins: "Does answering `request` require external tools, search or private data?",
		}},
		{ID: "is_sensitive", Q: question.NoulQuestion{
			Ins: "Does `request` involve money, legal, medical or safety consequences?",
		}},
	}
}

// All returns every preset by the name its upstream constructor carries, for
// tests and for a CLI that wants to list them. The map is rebuilt per call
// because Questions is a slice and a shared one would be mutable by a caller.
func All() map[string]question.Questions {
	return map[string]question.Questions{
		"triage":     TriageQuestions(),
		"email":      EmailQuestions(),
		"guard":      GuardQuestions(),
		"moderation": ModerationQuestions(),
		"router":     RouterQuestions(),
	}
}
