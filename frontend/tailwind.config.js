/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        // Medical / scientific palette
        brand: {
          50: '#eef6ff',
          100: '#d9ecff',
          200: '#bcdcff',
          300: '#8ec5ff',
          400: '#59a3ff',
          500: '#337eff',
          600: '#1e5ef5',
          700: '#1649d6',
          800: '#163cae',
          900: '#163789',
        },
        slate: {
          // Tailwind already provides slate; keep default.
        },
        risk: {
          low: '#10b981',
          moderate: '#f59e0b',
          high: '#ef4444',
        },
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
      },
      boxShadow: {
        card: '0 1px 3px 0 rgba(15, 23, 42, 0.06), 0 1px 2px -1px rgba(15, 23, 42, 0.08)',
      },
    },
  },
  plugins: [],
}
