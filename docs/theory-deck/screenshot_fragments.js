const puppeteer = require("puppeteer");
const path = require("path");
const fs = require("fs");

(async () => {
  const outDir = path.join(__dirname, "screenshots", "fragments");
  fs.mkdirSync(outDir, { recursive: true });

  const browser = await puppeteer.launch({ headless: "new", args: ["--no-sandbox"] });
  const page = await browser.newPage();
  page.on("pageerror", (err) => console.log("PAGEERROR:", err.message));
  await page.setViewport({ width: 1280, height: 720 });

  const fileUrl = "file://" + path.join(__dirname, "presentation.html") + "?export";
  await page.goto(fileUrl, { waitUntil: "networkidle0" });
  await page.evaluate(() => Reveal.configure({ transition: "none" }));

  // Navigate to slide-11 by index (find it)
  const targetId = process.argv[2] || "slide-11";
  const idx = await page.evaluate((tid) => {
    const slides = Reveal.getSlides();
    return slides.findIndex((s) => s.id === tid);
  }, targetId);
  console.log(targetId, "index:", idx);
  await page.evaluate((i) => Reveal.slide(i, 0, -1), idx); // horizontal index i, no vertical, no fragments shown
  await new Promise((r) => setTimeout(r, 400));

  // capture state with 0 fragments shown
  await page.screenshot({ path: path.join(outDir, "step-0.png") });
  console.log("captured step 0");

  for (let step = 1; step <= 5; step++) {
    await page.evaluate(() => Reveal.nextFragment());
    await new Promise((r) => setTimeout(r, 400));
    await page.screenshot({ path: path.join(outDir, `step-${step}.png`) });
    console.log("captured step", step);
  }

  await browser.close();
})();
