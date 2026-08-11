import { randomUUID } from "node:crypto";
import { mkdtemp, rm } from "node:fs/promises";
import { homedir, tmpdir } from "node:os";
import {
	basename,
	extname,
	isAbsolute,
	join,
	resolve,
} from "node:path";
import {
	createReadToolDefinition,
	type ExtensionAPI,
	type ExtensionContext,
} from "@earendil-works/pi-coding-agent";
import sharp, { type Sharp } from "sharp";
import { type Static, Type } from "typebox";

const CROP_STATUS_KEY = "focused-vision-crop";

const NormalizedBox = Type.Array(
	Type.Integer({ minimum: 0, maximum: 1000 }),
	{
		minItems: 4,
		maxItems: 4,
		description:
			"Optional crop region [x1,y1,x2,y2] in normalized 0-1000 coordinates. The crop is taken from the original image before Pi resizes it.",
	},
);

export const focusedReadSchema = Type.Object({
	path: Type.String({
		description: "Path to the file to read (relative or absolute)",
	}),
	offset: Type.Optional(
		Type.Number({ description: "Line number to start reading from (1-indexed)" }),
	),
	limit: Type.Optional(
		Type.Number({ description: "Maximum number of lines to read" }),
	),
	box: Type.Optional(NormalizedBox),
});

export type FocusedReadInput = Static<typeof focusedReadSchema>;

export interface FocusedReadOptions {
	enabled: boolean;
}

export interface PixelCrop {
	left: number;
	top: number;
	width: number;
	height: number;
	right: number;
	bottom: number;
}

interface CroppedImage {
	path: string;
	summary: string;
}

function normalizeInputPath(rawPath: string, cwd: string): string {
	let value = rawPath.trim();
	if (value.startsWith("@")) value = value.slice(1);
	if (value === "~") value = homedir();
	else if (value.startsWith("~/")) value = join(homedir(), value.slice(2));
	return isAbsolute(value) ? value : resolve(cwd, value);
}

function cleanStem(filePath: string): string {
	const extension = extname(filePath);
	const stem = basename(filePath, extension)
		.replace(/[^A-Za-z0-9._-]+/g, "-")
		.replace(/^-+|-+$/g, "")
		.slice(0, 80);
	return stem || "image";
}

function clamp(value: number, minimum: number, maximum: number): number {
	return Math.max(minimum, Math.min(maximum, value));
}

export function normalizedBoxToPixels(
	box: readonly number[],
	orientedWidth: number,
	orientedHeight: number,
): PixelCrop {
	if (box.length !== 4) {
		throw new Error("read box must contain exactly four coordinates");
	}
	if (!Number.isInteger(orientedWidth) || !Number.isInteger(orientedHeight)) {
		throw new Error("image dimensions must be integers");
	}
	if (orientedWidth <= 0 || orientedHeight <= 0) {
		throw new Error("image dimensions must be positive");
	}
	const coordinates = box.map((value) => Number(value));
	if (
		coordinates.some(
			(value) => !Number.isInteger(value) || value < 0 || value > 1000,
		)
	) {
		throw new Error("read box coordinates must be integers from 0 to 1000");
	}
	const [nx1, ny1, nx2, ny2] = coordinates as [
		number,
		number,
		number,
		number,
	];
	if (nx1 >= nx2 || ny1 >= ny2) {
		throw new Error("read box requires x1 < x2 and y1 < y2");
	}

	const left = clamp(Math.round((nx1 / 1000) * orientedWidth), 0, orientedWidth);
	const top = clamp(Math.round((ny1 / 1000) * orientedHeight), 0, orientedHeight);
	const right = clamp(Math.round((nx2 / 1000) * orientedWidth), 0, orientedWidth);
	const bottom = clamp(Math.round((ny2 / 1000) * orientedHeight), 0, orientedHeight);
	if (right <= left || bottom <= top) {
		throw new Error(
			`read box resolves to an empty crop on ${orientedWidth}x${orientedHeight}`,
		);
	}
	return {
		left,
		top,
		width: right - left,
		height: bottom - top,
		right,
		bottom,
	};
}

async function writeSharpWithAbort(
	pipeline: Sharp,
	outputPath: string,
	signal?: AbortSignal,
): Promise<void> {
	if (signal?.aborted) throw new Error("Operation aborted");
	await new Promise<void>((resolvePromise, rejectPromise) => {
		let settled = false;
		const cleanup = (): void => signal?.removeEventListener("abort", onAbort);
		const resolve = (): void => {
			if (settled) return;
			settled = true;
			cleanup();
			resolvePromise();
		};
		const reject = (error: unknown): void => {
			if (settled) return;
			settled = true;
			cleanup();
			rejectPromise(error instanceof Error ? error : new Error(String(error)));
		};
		const onAbort = (): void => {
			pipeline.destroy();
			reject(new Error("Operation aborted"));
		};
		signal?.addEventListener("abort", onAbort, { once: true });
		pipeline.toFile(outputPath).then(resolve, reject);
	});
}

export class FocusedReadRuntime {
	private tempRootPromise: Promise<string> | undefined;

	private async tempRoot(): Promise<string> {
		this.tempRootPromise ??= mkdtemp(join(tmpdir(), "pi-text-model-vision-"));
		return this.tempRootPromise;
	}

	private async temporaryCropPath(sourcePath: string): Promise<string> {
		return join(
			await this.tempRoot(),
			`${cleanStem(sourcePath)}-crop-${randomUUID()}.png`,
		);
	}

	async cropOriginal(
		sourcePath: string,
		box: readonly number[],
		signal?: AbortSignal,
	): Promise<CroppedImage> {
		if (signal?.aborted) throw new Error("Operation aborted");
		let metadata: Awaited<ReturnType<ReturnType<typeof sharp>["metadata"]>>;
		try {
			metadata = await sharp(sourcePath, { failOn: "error" }).metadata();
		} catch (error) {
			throw new Error(
				`无法从原图裁剪 ${sourcePath}：${error instanceof Error ? error.message : String(error)}`,
			);
		}
		if (signal?.aborted) throw new Error("Operation aborted");
		const orientedWidth = metadata.autoOrient?.width ?? metadata.width;
		const orientedHeight = metadata.autoOrient?.height ?? metadata.height;
		if (!orientedWidth || !orientedHeight) {
			throw new Error(`无法确定图片尺寸：${sourcePath}`);
		}
		const crop = normalizedBoxToPixels(box, orientedWidth, orientedHeight);
		const outputPath = await this.temporaryCropPath(sourcePath);
		const pipeline = sharp(sourcePath, { failOn: "error" })
			.autoOrient()
			.extract({
				left: crop.left,
				top: crop.top,
				width: crop.width,
				height: crop.height,
			})
			.png({ compressionLevel: 6 });
		try {
			await writeSharpWithAbort(pipeline, outputPath, signal);
		} catch (error) {
			await rm(outputPath, { force: true }).catch(() => undefined);
			throw error;
		}
		return {
			path: outputPath,
			summary: [
				`Cropped original image: ${sourcePath}`,
				`Box: [${box.join(",")}] → oriented pixel [${crop.left},${crop.top},${crop.right},${crop.bottom}]`,
				`Original oriented size: ${orientedWidth}x${orientedHeight}; crop: ${crop.width}x${crop.height}`,
				`Temporary crop path: ${outputPath}`,
			].join("\n"),
		};
	}

	async dispose(): Promise<void> {
		const rootPromise = this.tempRootPromise;
		this.tempRootPromise = undefined;
		if (!rootPromise) return;
		const root = await rootPromise.catch(() => undefined);
		if (root) await rm(root, { recursive: true, force: true }).catch(() => undefined);
	}
}

export function registerFocusedRead(
	pi: ExtensionAPI,
	options: FocusedReadOptions,
): FocusedReadRuntime {
	const runtime = new FocusedReadRuntime();
	if (!options.enabled) return runtime;

	pi.registerTool({
		name: "read",
		label: "read · text-model vision",
		description:
			"Read files with Pi's native read behavior. Text and image overviews delegate unchanged to Pi. For an image detail, pass box=[x1,y1,x2,y2] in normalized 0-1000 coordinates; the wrapper crops that region from the original, EXIF-oriented image before Pi resizes it, returns a temporary crop path for optional deeper focus, then routes the crop through the visual sidecar.",
		promptSnippet:
			"Read files natively; optionally crop an original-resolution image region with box=[x1,y1,x2,y2]",
		promptGuidelines: [
			"Use read instead of cat or sed for file contents; without box, read preserves Pi's native offset, limit, truncation, and image behavior.",
			"For image details, call read once without box for context, then call the same original path with the smallest useful normalized box from the visual observation; for another focus round, use the returned temporary crop path because new boxes are relative to the current crop.",
		],
		parameters: focusedReadSchema,
		async execute(toolCallId, params, signal, _onUpdate, ctx) {
			const nativeRead = createReadToolDefinition(ctx.cwd);
			if (!params.box) {
				return nativeRead.execute(
					toolCallId,
					{
						path: params.path,
						offset: params.offset,
						limit: params.limit,
					},
					signal,
					undefined,
					ctx,
				);
			}

			const sourcePath = normalizeInputPath(params.path, ctx.cwd);
			if (ctx.hasUI) ctx.ui.setStatus(CROP_STATUS_KEY, "focus: crop original");
			try {
				const cropped = await runtime.cropOriginal(sourcePath, params.box, signal);
				const result = await nativeRead.execute(
					toolCallId,
					{ path: cropped.path },
					signal,
					undefined,
					ctx,
				);
				return {
					content: [
						{ type: "text" as const, text: cropped.summary },
						...result.content,
					],
					details: result.details,
				};
			} finally {
				if (ctx.hasUI) ctx.ui.setStatus(CROP_STATUS_KEY, undefined);
			}
		},
	});

	return runtime;
}

export function clearFocusedReadStatus(ctx: ExtensionContext): void {
	if (ctx.hasUI) ctx.ui.setStatus(CROP_STATUS_KEY, undefined);
}
