/**
 * Proxies /api/* to your deployed FastAPI backend.
 * Vercel → Project → Settings → Environment Variables:
 *   BACKEND_URL = https://<your-service>.onrender.com   (no trailing slash)
 * Redeploy after adding or changing BACKEND_URL.
 */
function readBody(req) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    req.on("data", (c) => chunks.push(c));
    req.on("end", () => resolve(Buffer.concat(chunks)));
    req.on("error", reject);
  });
}

export default async function handler(req, res) {
  const backend = process.env.BACKEND_URL;
  if (!backend) {
    res.status(502).json({
      detail: "Set BACKEND_URL in Vercel to your API origin (e.g. https://your-app.onrender.com)",
    });
    return;
  }

  const base = backend.replace(/\/$/, "");
  const slug = req.query.slug;
  const parts = Array.isArray(slug) ? slug : slug != null ? [String(slug)] : [];
  const pathSuffix = parts.join("/");
  const qs = req.url?.includes("?") ? req.url.slice(req.url.indexOf("?")) : "";
  const targetUrl = `${base}/api/${pathSuffix}${qs}`;

  const fwd = {};
  const ct = req.headers["content-type"];
  if (ct) fwd["content-type"] = ct;

  const init = { method: req.method, headers: fwd };

  if (req.method !== "GET" && req.method !== "HEAD") {
    const body = await readBody(req);
    if (body.length) init.body = body;
  }

  let r;
  try {
    r = await fetch(targetUrl, init);
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    res.status(502).json({ detail: "Could not reach backend", error: msg });
    return;
  }

  const text = await r.text();
  const outCt = r.headers.get("content-type") || "application/json";
  res.status(r.status).setHeader("Content-Type", outCt);
  res.send(text);
}
