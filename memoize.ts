import crypto from "crypto";
import sqlite3 from "sqlite3";
import { open } from "sqlite";

type JsonSerializable =
  | null
  | boolean
  | number
  | string
  | JsonSerializable[]
  | { [key: string]: JsonSerializable };

type SerializeableFn = (
  ...args: any[]
) => JsonSerializable | Promise<JsonSerializable> | unknown;

interface MemoizeOptions<T extends SerializeableFn> {
  cacheKey?: (...args: Parameters<T>) => string | null;
  maxAge?: number;
  dbPath: string;
}

export default function memoize<T extends SerializeableFn>(
  options: MemoizeOptions<T>,
  fn: T
): (...args: Parameters<T>) => Promise<ReturnType<T>> {
  const cacheKeyFn =
    options?.cacheKey ||
    ((...args) => JSON.stringify({ ...args, fn: fn.toString() }));
  const dbPath = options?.dbPath;

  // Initialize SQLite DB and create table if it doesn't exist
  const initializeDb = async () => {
    const db = await open({ filename: dbPath, driver: sqlite3.Database });
    await db.exec(`
      CREATE TABLE IF NOT EXISTS memoized_responses (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        cache_key TEXT UNIQUE,
        response TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
      );
    `);
    await db.close();
  };

  initializeDb();

  const memoizedFn = async (...args: Parameters<T>): Promise<ReturnType<T>> => {
    let cacheKey = cacheKeyFn(...args);

    if (cacheKey === null) {
      return fn(...args) as ReturnType<T>;
    }

    if (cacheKey.length > 1000) {
      cacheKey = crypto.createHash("sha256").update(cacheKey).digest("hex");
    }

    const currentTime = new Date();

    let memoizedResponse: any;
    const db = await open({ filename: dbPath, driver: sqlite3.Database });

    try {
      memoizedResponse = await db.get(
        "SELECT * FROM memoized_responses WHERE cache_key = ?",
        [cacheKey]
      );
    } catch (error) {
      console.error("Error fetching memoized response:", error);
      memoizedResponse = null;
    }

    if (
      memoizedResponse &&
      (!options?.maxAge ||
        currentTime.getTime() -
          new Date(memoizedResponse.created_at).getTime() <=
          options.maxAge)
    ) {
      await db.close();
      return JSON.parse(memoizedResponse.response) as ReturnType<T>;
    } else {
      const result = await fn(...args);
      const typedResult = result == null ? "null" : JSON.stringify(result);

      try {
        await db.run(
          "INSERT OR REPLACE INTO memoized_responses (cache_key, response) VALUES (?, ?)",
          [cacheKey, typedResult]
        );
      } catch (error) {
        console.error("Error storing memoized response:", error);
      }

      await db.close();
      return result as ReturnType<T>;
    }
  };

  return memoizedFn;
}
