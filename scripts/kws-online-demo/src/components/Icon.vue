<script setup lang="ts">
defineProps<{ name: string }>();
const paths: Record<string, string> = {
  mic: "M9 5a3 3 0 0 1 6 0v7a3 3 0 0 1-6 0V5ZM5 10v2a7 7 0 0 0 14 0v-2M12 19v3m-4 0h8",
  stop: "M7 7h10v10H7Z",
  arrow: "M7 17 17 7M7 7h10v10",
  lock: "M5 10h14v11H5ZM8 10V7a4 4 0 0 1 8 0v3",
  check: "m5 12 4 4L19 6",
  download: "M12 3v12m-5-5 5 5 5-5M5 16v5h14v-5",
  file: "M14 2H6v20h12V6l-4-4ZM14 2v5h5M9 13h6m-6 4h6",
  trash: "M3 6h18M9 6V3h6v3M6 6l1 15h10l1-15M10 10v7m4-7v7",
  refresh: "M20 10a8 8 0 1 0-1 8M20 4v6h-6",
  close: "m6 6 12 12M6 18 18 6",
  chevron: "m8 10 4 4 4-4",
  headphones: "M3 14v-3a9 9 0 0 1 18 0v3M3 13h4v8H3v-8Zm14 0h4v8h-4v-8Z",
};
</script>
<template>
  <svg
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    stroke-width="1.65"
    stroke-linecap="round"
    stroke-linejoin="round"
    aria-hidden="true"
  >
    <path :d="paths[name] || paths.mic" />
  </svg>
</template>
