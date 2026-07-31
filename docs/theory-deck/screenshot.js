const puppeteer = require("puppeteer");
const path = require("path");
const fs = require("fs");

(async () => {
  const outDir = path.join(__dirname, "screenshots", "run1");
  fs.mkdirSync(outDir, { recursive: true });

  const browser = await puppeteer.launch({
    headless: "new",
    args: ["--no-sandbox", "--disable-setuid-sandbox"],
  });
  const page = await browser.newPage();
  await page.setViewport({ width: 1280, height: 720 });

  const fileUrl = "file://" + path.join(__dirname, "presentation.html") + "?export";
  await page.goto(fileUrl, { waitUntil: "networkidle0" });
  await page.evaluate(() => Reveal.configure({ transition: "none" }));

  const total = await page.evaluate(() => Reveal.getTotalSlides());
  console.log("total slides:", total);

  for (let i = 0; i < total; i++) {
    await page.evaluate((idx) => Reveal.slide(idx), i);
    await new Promise((r) => setTimeout(r, 800));
    const num = String(i + 1).padStart(2, "0");
    await page.screenshot({ path: path.join(outDir, `slide-${num}.png`) });
    console.log("captured slide", num);
  }

  await browser.close();
})();
