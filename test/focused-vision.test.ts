import assert from "node:assert/strict";
import { access, mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import type { ImageContent, Usage } from "@earendil-works/pi-ai";
import sharp from "sharp";
import type {
	ExtensionAPI,
	ExtensionContext,
} from "@earendil-works/pi-coding-agent";
import {
	combineUsage,
	loadFocusedVisionConfig,
	parseVisualReport,
	registerFocusedVision,
	selectEvenly,
	type FocusedVisionConfig,
	type VisualReport,
} from "../index.ts";
import { normalizedBoxToPixels } from "../focused-read.ts";

const IMAGE: ImageContent = {
	type: "image",
	data: Buffer.from("fake-image").toString("base64"),
	mimeType: "image/png",
};

const REPORT: VisualReport = {
	conclusion: "可见一个红色错误弹窗。",
	visibleText: ["Connection failed"],
	spatialSummary: "弹窗位于图片中央。",
	clarity: "partial",
	focusRequired: true,
	uncertainRegions: [
		{
			label: "弹窗底部小字",
			bbox: [200, 600, 800, 760],
			legibility: "uncertain",
			reason: "too_small",
			observedFragment: "Error ...",
			affectsAnswer: true,
		},
	],
};

function reportToolCall(report: VisualReport = REPORT): unknown[] {
	return [
		{
			type: "toolCall",
			id: "vision-report-1",
			name: "report_visual_observation",
			arguments: report,
		},
	];
}

const USAGE: Usage = {
	input: 10,
	output: 5,
	cacheRead: 1,
	cacheWrite: 0,
	reasoning: 2,
	totalTokens: 16,
	cost: {
		input: 0.1,
		output: 0.2,
		cacheRead: 0.01,
		cacheWrite: 0,
		total: 0.31,
	},
};

const CONFIG: FocusedVisionConfig = {
	enabled: true,
	routeInputImages: true,
	routeToolResultImages: true,
	overrideRead: true,
	provider: "opencode-go",
	model: "mimo-v2.5",
	maxImages: 16,
	maxTokens: 4096,
	timeoutMs: 180_000,
	cacheEntries: 32,
	reasoningEffort: "low",
};

type EventHandler = (...args: any[]) => unknown;

function harness(options?: {
	mainSupportsImages?: boolean;
	branch?: unknown[];
	completeContents?: unknown[][];
	hasUI?: boolean;
	sidecarAvailable?: boolean;
}) {
	const handlers = new Map<string, EventHandler[]>();
	const commands = new Map<string, unknown>();
	const tools = new Map<string, any>();
	let completeCalls = 0;
	let lastCompleteArgs: unknown[] | undefined;
	const notifications: string[] = [];
	const targetModel = {
		id: "mimo-v2.5",
		provider: "opencode-go",
		api: "openai-completions",
		input: ["text", "image"],
	};
	const mainModel = {
		id: options?.mainSupportsImages ? "vision-main" : "deepseek-v4-flash",
		provider: "opencode-go",
		api: "openai-completions",
		input: options?.mainSupportsImages ? ["text", "image"] : ["text"],
	};
	const pi = {
		on(name: string, handler: EventHandler) {
			handlers.set(name, [...(handlers.get(name) ?? []), handler]);
		},
		registerTool(tool: any) {
			tools.set(tool.name, tool);
		},
		registerCommand(name: string, command: unknown) {
			commands.set(name, command);
		},
		getAllTools() {
			return [...tools.values()];
		},
		getActiveTools() {
			return [...tools.keys()];
		},
	} as unknown as ExtensionAPI;
	const ctx = {
		cwd: process.cwd(),
		model: mainModel,
		signal: undefined,
		hasUI: options?.hasUI ?? false,
		ui: {
			setStatus() {},
			notify(message: string) {
				notifications.push(message);
			},
		},
		modelRegistry: {
			find(provider: string, model: string) {
				return options?.sidecarAvailable !== false &&
					provider === "opencode-go" &&
					model === "mimo-v2.5"
					? targetModel
					: undefined;
			},
			hasConfiguredAuth() {
				return true;
			},
			async complete(...args: unknown[]) {
				completeCalls++;
				lastCompleteArgs = args;
				return {
					content:
						options?.completeContents?.[completeCalls - 1] ?? reportToolCall(),
					stopReason: "toolUse",
					usage: USAGE,
				};
			},
		},
		sessionManager: {
			getBranch() {
				return options?.branch ?? [];
			},
		},
	} as unknown as ExtensionContext;

	registerFocusedVision(pi, CONFIG);
	return {
		ctx,
		handler(name: string): EventHandler {
			const handler = handlers.get(name)?.[0];
			assert.ok(handler, `missing ${name} handler`);
			return handler;
		},
		get completeCalls() {
			return completeCalls;
		},
		get lastCompleteArgs() {
			return lastCompleteArgs;
		},
		tool(name: string): any {
			const tool = tools.get(name);
			assert.ok(tool, `missing ${name} tool`);
			return tool;
		},
		get toolNames(): string[] {
			return [...tools.keys()];
		},
		hasHandler(name: string): boolean {
			return handlers.has(name);
		},
		get commandNames(): string[] {
			return [...commands.keys()];
		},
		get notifications(): string[] {
			return notifications;
		},
	};
}

test("routes direct image input through Mimo as a hidden context message", async () => {
	const h = harness();
	const event = {
		type: "before_agent_start",
		prompt: "这个截图报了什么错？",
		images: [IMAGE],
		systemPrompt: "system",
		systemPromptOptions: {},
	};
	const result = (await h.handler("before_agent_start")(event, h.ctx)) as {
		message: {
			customType: string;
			content: string;
			display: boolean;
		};
	};

	assert.equal(result.message.customType, "focused-vision-observation");
	assert.equal(result.message.display, false);
	assert.match(result.message.content, /focused_vision_observation/);
	assert.match(result.message.content, /clarity="partial"/);
	assert.match(result.message.content, /弹窗底部小字/);
	assert.match(result.message.content, /红色错误弹窗/);
	assert.doesNotMatch(result.message.content, /不可信|不是指令|需要放大/);
	assert.equal(h.completeCalls, 1);

	const [, context, requestOptions] = h.lastCompleteArgs!;
	const requestContent = (context as any).messages[0].content;
	assert.ok(requestContent.some((block: any) => block.type === "image"));
	assert.equal((context as any).tools[0].name, "report_visual_observation");
	assert.deepEqual((requestOptions as any).toolChoice, {
		type: "function",
		function: { name: "report_visual_observation" },
	});
});

test("deduplicates identical input observations in the session cache", async () => {
	const h = harness();
	const event = {
		type: "before_agent_start",
		prompt: "描述图片",
		images: [IMAGE],
		systemPrompt: "system",
		systemPromptOptions: {},
	};
	await h.handler("before_agent_start")(event, h.ctx);
	const second = (await h.handler("before_agent_start")(event, h.ctx)) as {
		message: { content: string };
	};

	assert.equal(h.completeCalls, 1);
	assert.match(second.message.content, /cached="true"/);
});

test("retries once when Mimo does not return the forced report tool", async () => {
	const h = harness({
		completeContents: [
			[{ type: "text", text: "free-form response" }],
			reportToolCall(),
		],
	});
	const result = (await h.handler("before_agent_start")(
		{
			type: "before_agent_start",
			prompt: "描述图片",
			images: [IMAGE],
			systemPrompt: "system",
			systemPromptOptions: {},
		},
		h.ctx,
	)) as { message: { content: string; details: { usage: Usage } } };
	assert.equal(h.completeCalls, 2);
	assert.match(result.message.content, /"clarity": "partial"/);
	assert.equal(result.message.details.usage.input, USAGE.input * 2);
});

test("rejects structurally contradictory visual reports", () => {
	assert.throws(
		() =>
			parseVisualReport(
				reportToolCall({
					...REPORT,
					clarity: "clear",
					focusRequired: false,
				}),
			),
		/clarity=clear/,
	);
});

test("routes image tool results and carries nested usage", async () => {
	const h = harness({
		branch: [
			{
				type: "message",
				message: {
					role: "user",
					content: [{ type: "text", text: "读出截图里的错误" }],
				},
			},
		],
	});
	const event = {
		type: "tool_result",
		toolName: "read",
		toolCallId: "call-1",
		input: { path: "/tmp/error.png" },
		content: [
			{
				type: "text",
				text: "Read image file [image/png]\n[Current model does not support images. The image will be omitted from this request.]",
			},
			IMAGE,
		],
		details: undefined,
		isError: false,
	};
	const result = (await h.handler("tool_result")(event, h.ctx)) as {
		content: Array<{ type: string; text?: string }>;
		usage: Usage;
	};

	assert.ok(result.content.some((block) => block.type === "image"));
	assert.match(result.content.at(-1)?.text ?? "", /红色错误弹窗/);
	assert.doesNotMatch(
		result.content.map((block) => block.text ?? "").join("\n"),
		/Current model does not support images/,
	);
	assert.deepEqual(result.usage, USAGE);
	assert.equal(h.completeCalls, 1);
});

test("does not route when the current main model already accepts images", async () => {
	const h = harness({ mainSupportsImages: true });
	const result = await h.handler("before_agent_start")(
		{
			type: "before_agent_start",
			prompt: "描述图片",
			images: [IMAGE],
			systemPrompt: "system",
			systemPromptOptions: {},
		},
		h.ctx,
	);
	assert.equal(result, undefined);
	assert.equal(h.completeCalls, 0);
});

test("selectEvenly keeps endpoints and combineUsage preserves both calls", () => {
	assert.deepEqual(
		selectEvenly([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 4).map(
			(item) => item.index,
		),
		[0, 3, 6, 9],
	);
	const combined = combineUsage(USAGE, USAGE)!;
	assert.equal(combined.input, 20);
	assert.equal(combined.reasoning, 4);
	assert.equal(combined.cost.total, 0.62);
});

test("registers only the read override, without a skill or command", () => {
	const h = harness();
	assert.deepEqual(h.toolNames, ["read"]);
	assert.deepEqual(h.commandNames, []);
	assert.equal(h.hasHandler("resources_discover"), false);
	const read = h.tool("read");
	assert.deepEqual(Object.keys(read.parameters.properties), [
		"path",
		"offset",
		"limit",
		"box",
	]);
	assert.deepEqual(normalizedBoxToPixels([500, 0, 1000, 1000], 100, 50), {
		left: 50,
		top: 0,
		width: 50,
		height: 50,
		right: 100,
		bottom: 50,
	});
});

test("warns text-only interactive sessions when opencode-go sidecar is missing", async () => {
	const h = harness({ hasUI: true, sidecarAvailable: false });
	await h.handler("session_start")({ type: "session_start" }, h.ctx);
	assert.equal(h.notifications.length, 1);
	assert.match(h.notifications[0]!, /^Text-model vision:/);
	assert.match(h.notifications[0]!, /opencode-go\/mimo-v2\.5/);
	assert.match(h.notifications[0]!, /pi --list-models opencode-go/);
});

test("the read override preserves Pi native text behavior", async () => {
	const directory = await mkdtemp(join(tmpdir(), "focused-read-test-"));
	try {
		const path = join(directory, "sample.txt");
		await writeFile(path, "alpha\nbeta\ngamma\n", "utf8");
		const h = harness();
		const result = await h.tool("read").execute(
			"read-1",
			{ path, offset: 2, limit: 1 },
			undefined,
			undefined,
			h.ctx,
		);
		assert.equal(result.content[0]?.type, "text");
		assert.match(result.content[0]?.text ?? "", /^beta/);
		assert.match(result.content[0]?.text ?? "", /more lines in file/);
	} finally {
		await rm(directory, { recursive: true, force: true });
	}
});

test("box crops the EXIF-oriented original before native read", async () => {
	const directory = await mkdtemp(join(tmpdir(), "focused-crop-test-"));
	const source = join(directory, "split-oriented.jpg");
	const h = harness();
	let temporaryCropPath: string | undefined;
	try {
		await sharp(
			Buffer.from(
				'<svg width="100" height="50"><rect width="50" height="50" fill="red"/><rect x="50" width="50" height="50" fill="blue"/></svg>',
			),
		)
			.jpeg({ quality: 100, chromaSubsampling: "4:4:4" })
			.withMetadata({ orientation: 6 })
			.toFile(source);
		const result = await h.tool("read").execute(
			"read-crop-1",
			{ path: source, box: [0, 500, 1000, 1000] },
			undefined,
			undefined,
			h.ctx,
		);
		assert.match(result.content[0]?.text ?? "", /Original oriented size: 50x100; crop: 50x50/);
		const pathMatch = /Temporary crop path: (.+)/.exec(
			result.content[0]?.text ?? "",
		);
		assert.ok(pathMatch?.[1]);
		temporaryCropPath = pathMatch[1];
		const image = result.content.find((block: any) => block.type === "image");
		assert.ok(image);
		const cropped = sharp(Buffer.from(image.data, "base64"));
		const metadata = await cropped.metadata();
		const stats = await cropped.stats();
		assert.equal(metadata.width, 50);
		assert.equal(metadata.height, 50);
		assert.ok(stats.channels[2]!.mean > 220);
		assert.ok(stats.channels[0]!.mean < 30);
	} finally {
		await h.handler("session_shutdown")(
			{ type: "session_shutdown" },
			h.ctx,
		);
		if (temporaryCropPath) await assert.rejects(access(temporaryCropPath));
		await rm(directory, { recursive: true, force: true });
	}
});

test("environment configuration is bounded and preserves legacy aliases", () => {
	const config = loadFocusedVisionConfig({
		PI_TEXT_MODEL_VISION_DISABLED: "1",
		PI_TEXT_MODEL_VISION_MAX_IMAGES: "999",
		PI_TEXT_MODEL_VISION_MAX_TOKENS: "1",
		PI_TEXT_MODEL_VISION_REASONING_EFFORT: "high",
	});
	assert.equal(config.enabled, false);
	assert.equal(config.maxImages, 32);
	assert.equal(config.maxTokens, 256);
	assert.equal(config.reasoningEffort, "high");
	assert.equal(config.overrideRead, true);

	const legacy = loadFocusedVisionConfig({
		PI_FOCUSED_VISION_INPUT: "0",
		PI_VISION_ROUTER_TOOL_RESULTS: "0",
	});
	assert.equal(legacy.routeInputImages, false);
	assert.equal(legacy.routeToolResultImages, false);
});
