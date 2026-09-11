/**
 * Converts pasted rich-text HTML (from a webpage, ChatGPT, Google Docs,
 * Notion, Word, etc.) into plain text that preserves bullet and numbered
 * list structure — a real "•" bullet character for unordered items,
 * "1. "/"2. " for ordered ones, with two-space indentation per nesting level.
 * "•" (not a hyphen) deliberately, so a <textarea> — which can only ever
 * hold plain text and has no way to render an actual HTML list marker —
 * still shows something that reads as a real bullet, not a dash.
 *
 * Why this exists: pasting rich content into a plain <textarea> always goes
 * through the *browser's own* HTML-to-plain-text conversion first — and
 * that conversion is inconsistent across browsers and sources. Some drop
 * the bullet/number markers entirely, some keep them, some collapse nested
 * lists flat. Intercepting the paste and doing this conversion ourselves
 * means the same source list always produces the same, correct text,
 * regardless of what browser or where the content was copied from.
 */

const BLOCK_TAGS = new Set([
  "p",
  "div",
  "h1",
  "h2",
  "h3",
  "h4",
  "h5",
  "h6",
  "blockquote",
  "tr",
  "table",
  "section",
  "article",
]);

interface ListContext {
  type: "ul" | "ol";
  counter: number;
}

export function htmlToStructuredText(html: string): string {
  const doc = new DOMParser().parseFromString(html, "text/html");
  const lines: string[] = [];
  let currentLine = "";
  // Held separately from currentLine, and never passed through trim() —
  // trimming currentLine at flush time is what makes list-item indentation
  // safe to build (accidental leading/trailing whitespace from collapsed
  // text nodes doesn't leak into the output), but that same trim() would
  // just as happily eat an intentional nested-list indent if it were part
  // of currentLine instead of tracked on the side like this.
  let pendingPrefix = "";

  function flushLine() {
    const body = currentLine.trim();
    if (body || pendingPrefix) lines.push(pendingPrefix + body);
    currentLine = "";
    pendingPrefix = "";
  }

  function breakParagraph() {
    flushLine();
    if (lines.length > 0 && lines[lines.length - 1] !== "") {
      lines.push("");
    }
  }

  function walk(node: ChildNode, listStack: ListContext[]) {
    if (node.nodeType === Node.TEXT_NODE) {
      const text = (node.textContent || "").replace(/\s+/g, " ");
      if (text) currentLine += text;
      return;
    }
    if (node.nodeType !== Node.ELEMENT_NODE) return;

    const el = node as HTMLElement;
    const tag = el.tagName.toLowerCase();

    if (tag === "script" || tag === "style") return;

    if (tag === "ul" || tag === "ol") {
      flushLine();
      const nextStack = [...listStack, { type: tag as "ul" | "ol", counter: 1 }];
      Array.from(el.childNodes).forEach((child) => walk(child, nextStack));
      breakParagraph();
      return;
    }

    if (tag === "li") {
      flushLine();
      const depth = listStack.length;
      const top = listStack[depth - 1];
      const indent = "  ".repeat(Math.max(depth - 1, 0));
      const prefix = top?.type === "ol" ? `${top.counter}. ` : "• ";
      if (top?.type === "ol") top.counter += 1;
      pendingPrefix = indent + prefix;
      Array.from(el.childNodes).forEach((child) => walk(child, listStack));
      flushLine();
      return;
    }

    if (tag === "br") {
      flushLine();
      return;
    }

    if (BLOCK_TAGS.has(tag)) {
      breakParagraph();
      Array.from(el.childNodes).forEach((child) => walk(child, listStack));
      breakParagraph();
      return;
    }

    // Inline formatting (span, b, strong, i, em, a, ...) — keep its text,
    // drop the formatting itself; not what this app's plain-text input needs.
    Array.from(el.childNodes).forEach((child) => walk(child, listStack));
  }

  Array.from(doc.body.childNodes).forEach((child) => walk(child, []));
  flushLine();

  return lines.join("\n").replace(/\n{3,}/g, "\n\n").trim();
}

/** True if the HTML actually contains a list — used to decide whether
 * intercepting the paste is worth it at all (see the onPaste handler). */
export function containsList(html: string): boolean {
  return /<\s*(ul|ol)\b/i.test(html);
}
