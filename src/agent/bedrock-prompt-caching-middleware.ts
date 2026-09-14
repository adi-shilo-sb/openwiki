import { ChatBedrockConverse } from "@langchain/aws";
import { createMiddleware } from "langchain";
import { resolveBedrockCacheTtl } from "../config/constants.js";

/**
 * Creates middleware that asks Bedrock to cache each request's stable prefix.
 *
 * `ChatBedrockConverse` translates `cache_control` into Converse `cachePoint`
 * blocks after the system prompt, the tool definitions, and the final message.
 * Without it no cache point is emitted, so every call reprocesses the whole
 * prefix at full input price.
 *
 * LangChain's own `bedrockPromptCachingMiddleware` is not used because it
 * decides cache capability by substring-matching the model id against
 * `anthropic.claude` and `amazon.nova`. An application inference profile ARN
 * names a profile rather than a family, so it matches neither and silently
 * loses caching, even though Bedrock caches such profiles exactly as it caches
 * system profiles.
 *
 * The TTL is resolved once here rather than per request, so an invalid
 * `OPENWIKI_BEDROCK_CACHE_TTL` fails while the agent is being built.
 *
 * @returns LangChain middleware that sets `cache_control` on Bedrock requests.
 */
export function createBedrockPromptCachingMiddleware() {
  const ttl = resolveBedrockCacheTtl();

  return createMiddleware({
    name: "OpenWikiBedrockPromptCaching",
    wrapModelCall: (request, handler) =>
      ttl && request.model instanceof ChatBedrockConverse
        ? handler({
            ...request,
            modelSettings: {
              ...request.modelSettings,
              cache_control: { type: "ephemeral", ttl },
            },
          })
        : handler(request),
  });
}
