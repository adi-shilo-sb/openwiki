import { ChatBedrockConverse } from "@langchain/aws";
import { afterEach, beforeEach, describe, expect, test } from "vitest";
import { createBedrockPromptCachingMiddleware } from "../../src/agent/bedrock-prompt-caching-middleware.ts";

type CachingRequest = {
  model: unknown;
  modelSettings?: Record<string, unknown>;
};

type CachingMiddleware = {
  wrapModelCall?: (
    request: CachingRequest,
    handler: (next: CachingRequest) => Promise<CachingRequest>,
  ) => Promise<CachingRequest>;
};

const originalTtl = process.env.OPENWIKI_BEDROCK_CACHE_TTL;

beforeEach(() => {
  delete process.env.OPENWIKI_BEDROCK_CACHE_TTL;
});

afterEach(() => {
  if (originalTtl === undefined) {
    delete process.env.OPENWIKI_BEDROCK_CACHE_TTL;
  } else {
    process.env.OPENWIKI_BEDROCK_CACHE_TTL = originalTtl;
  }
});

function createBedrockModel(): ChatBedrockConverse {
  return new ChatBedrockConverse({
    model: "us.anthropic.claude-sonnet-5",
    region: "us-east-1",
    credentials: { accessKeyId: "test-access", secretAccessKey: "test-secret" },
  });
}

async function wrap(request: CachingRequest): Promise<CachingRequest> {
  const { wrapModelCall } =
    createBedrockPromptCachingMiddleware() as unknown as CachingMiddleware;

  if (!wrapModelCall) {
    throw new Error("Expected the Bedrock prompt-caching model-call wrapper.");
  }

  return wrapModelCall(request, (next) => Promise.resolve(next));
}

describe("createBedrockPromptCachingMiddleware", () => {
  test("requests a 5m ephemeral cache point for Bedrock Converse models", async () => {
    const forwarded = await wrap({ model: createBedrockModel() });

    expect(forwarded.modelSettings).toEqual({
      cache_control: { type: "ephemeral", ttl: "5m" },
    });
  });

  test("honors the configured TTL", async () => {
    process.env.OPENWIKI_BEDROCK_CACHE_TTL = "1h";

    const forwarded = await wrap({ model: createBedrockModel() });

    expect(forwarded.modelSettings).toEqual({
      cache_control: { type: "ephemeral", ttl: "1h" },
    });
  });

  test("forwards the request unchanged when caching is turned off", async () => {
    process.env.OPENWIKI_BEDROCK_CACHE_TTL = "off";
    const request = { model: createBedrockModel() };

    const forwarded = await wrap(request);

    expect(forwarded).toBe(request);
    expect(forwarded.modelSettings).toBeUndefined();
  });

  test("rejects an invalid TTL before the agent runs", () => {
    process.env.OPENWIKI_BEDROCK_CACHE_TTL = "10m";

    expect(() => createBedrockPromptCachingMiddleware()).toThrow(
      /OPENWIKI_BEDROCK_CACHE_TTL/u,
    );
  });

  test("forwards the request unchanged for every other provider", async () => {
    const request = { model: { _llmType: () => "openai" } };

    const forwarded = await wrap(request);

    expect(forwarded).toBe(request);
    expect(forwarded.modelSettings).toBeUndefined();
  });

  test("preserves model settings contributed by other middleware", async () => {
    const forwarded = await wrap({
      model: createBedrockModel(),
      modelSettings: { strict: true },
    });

    expect(forwarded.modelSettings).toEqual({
      strict: true,
      cache_control: { type: "ephemeral", ttl: "5m" },
    });
  });
});
