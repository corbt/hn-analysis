import Database from "better-sqlite3";
import { open } from "sqlite";
import axios from "axios";

const BASE_URL = "https://hacker-news.firebaseio.com/v0";
const BATCH_SIZE = 500;
const NUM_WORKERS = 50;
const REPORTING_INTERVAL = 1000;

const db = new Database("./data/items.db");
db.pragma("journal_mode = WAL");

db.prepare(
  "CREATE TABLE IF NOT EXISTS items (id INTEGER PRIMARY KEY, json TEXT)"
).run();

const ax = axios.create();

console.log("Fetching already pulled ids...");
let alreadyPulledIds: number[] = db
  .prepare("SELECT id FROM items")
  .all()
  .map((row) => row.id)
  .sort((a, b) => a - b);

console.log("Fetching max id...");
const maxId: number = await ax
  .get(`${BASE_URL}/maxitem.json`)
  .then((res) => res.data);

console.log("Max id:", maxId.toLocaleString());

const idsToFetch = (function* generateIdsToFetch() {
  let totalProcessed = alreadyPulledIds.length;
  let lastReportTime = Date.now();
  let listPointer = alreadyPulledIds.length - 1;

  for (let id = maxId; id >= 0; id--) {
    while (listPointer >= 0 && alreadyPulledIds[listPointer] > id) {
      listPointer--;
    }

    if (listPointer >= 0 && alreadyPulledIds[listPointer] === id) {
      listPointer--;
    } else {
      yield id;
      totalProcessed++;

      if (totalProcessed % REPORTING_INTERVAL === 0) {
        const currentTime = Date.now();
        const elapsedTime = (currentTime - lastReportTime) / 1000;
        const itemsPerSecond = REPORTING_INTERVAL / elapsedTime;
        const remainingTime = (maxId - totalProcessed) / itemsPerSecond;
        const etaHours = Math.floor(remainingTime / 3600);
        const etaMinutes = Math.floor((remainingTime % 3600) / 60);
        const etaSeconds = Math.floor(remainingTime % 60);
        const etaString = `${etaHours.toString().padStart(2, "0")}:${etaMinutes
          .toString()
          .padStart(2, "0")}:${etaSeconds.toString().padStart(2, "0")}`;

        console.log(
          `${totalProcessed.toLocaleString()}/${maxId.toLocaleString()} | ${(
            (totalProcessed / maxId) *
            100
          ).toFixed(2)}% | ${itemsPerSecond.toFixed(
            2
          )} items/s | ETA: ${etaString}`
        );

        lastReportTime = currentTime;
      }
    }
  }
})();

const worker = async () => {
  let batch: Array<[number, string]> = [];

  const persistBatch = async () => {
    if (batch.length > 0) {
      db.prepare(
        `INSERT INTO items (id, json) VALUES ${batch
          .map(() => "(?, ?)")
          .join(", ")}`
      ).run(batch.flat());
      // await insertStmt.run(batch.flat());
      // await insertStmt.finalize();
      batch = [];
    }
  };

  for (const id of idsToFetch) {
    try {
      const json = await ax
        .get(`${BASE_URL}/item/${id}.json`)
        .then((res) => res.data ?? { id, deleted: true });

      batch.push([json.id, JSON.stringify(json)]);

      if (batch.length >= BATCH_SIZE) await persistBatch();
    } catch (e) {
      console.error("Error fetching", id, e);
    }
  }

  // Insert any remaining items in the batch
  await persistBatch();
};

console.log("Spawning workers...");
const workers = Array.from({ length: NUM_WORKERS }, () => worker());

await Promise.all(workers);

console.log("Closing database...");
await db.close();

console.log("Done!");
