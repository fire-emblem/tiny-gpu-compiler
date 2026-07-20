import { Instruction, TargetArch, XCORE1000_MNEMONIC_CATEGORIES, XCore1000InstCategory } from '../compiler/types';

interface BinaryViewProps {
  instructions: Instruction[];
  highlightAddr?: number;
  target?: TargetArch;
  assembly?: string;
}

/** Color for xcore1000 instruction categories */
const CATEGORY_COLORS: Record<XCore1000InstCategory, string> = {
  [XCore1000InstCategory.INTEGER_ARITH]: '#e06c75',   // red
  [XCore1000InstCategory.BITWISE]: '#c678dd',         // purple
  [XCore1000InstCategory.FLOAT_ARITH]: '#61afef',     // blue
  [XCore1000InstCategory.CONVERSION]: '#56b6c2',      // cyan
  [XCore1000InstCategory.COMPARISON]: '#d19a66',      // orange
  [XCore1000InstCategory.SELECT]: '#e5c07b',          // yellow
  [XCore1000InstCategory.MEMORY_GLOBAL]: '#98c379',   // green
  [XCore1000InstCategory.MEMORY_SHARED]: '#56b6c2',   // cyan
  [XCore1000InstCategory.ATOMIC]: '#c678dd',          // purple
  [XCore1000InstCategory.BARRIER]: '#e06c75',         // red
  [XCore1000InstCategory.CONTROL_FLOW]: '#d19a66',    // orange
  [XCore1000InstCategory.WARP]: '#61afef',            // blue
  [XCore1000InstCategory.SPECIAL]: '#5c6370',         // gray
};

function getInstructionColor(mnemonic: string): string {
  const category = XCORE1000_MNEMONIC_CATEGORIES[mnemonic];
  return category ? CATEGORY_COLORS[category] : '#e0e0e0';
}

/** xcore1000 assembly view with instruction category color coding */
function XCore1000AssemblyView({ assembly }: { assembly: string }) {
  const lines = assembly.split('\n');
  return (
    <div style={{ overflow: 'auto', fontSize: '12px', fontFamily: 'monospace' }}>
      <table style={{ width: '100%', borderCollapse: 'collapse' }}>
        <thead>
          <tr style={{ background: '#111', color: '#888', textAlign: 'left' }}>
            <th style={{ padding: '6px 8px', width: '40px' }}>#</th>
            <th style={{ padding: '6px 8px' }}>xcore1000 Assembly</th>
          </tr>
        </thead>
        <tbody>
          {lines.map((line, i) => {
            const trimmed = line.trim();
            const mnemonic = trimmed.split(/\s+/)[0]?.replace(/;.*$/, '') || '';
            const color = getInstructionColor(mnemonic);
            const isLabel = trimmed.startsWith('.') || trimmed.startsWith('#') || trimmed.startsWith('//');
            const isComment = trimmed.startsWith(';');
            const isMeta = trimmed.startsWith('.macahca') || trimmed.startsWith('.end_macahca');

            return (
              <tr key={i} style={{ background: isMeta ? '#1a2a1a' : 'transparent' }}>
                <td style={{ padding: '2px 8px', color: '#555', fontSize: '11px' }}>{i + 1}</td>
                <td style={{
                  padding: '2px 8px',
                  color: isLabel ? '#61afef' : isComment ? '#5c6370' : isMeta ? '#56b6c2' : color,
                  fontWeight: isMeta ? 700 : 400,
                }}>
                  {line}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

/** Main BinaryView: shows tiny-gpu 16-bit or xcore1000 assembly */
export function BinaryView({ instructions, highlightAddr, target, assembly }: BinaryViewProps) {
  // xcore1000 mode: show assembly with color coding
  if (target === 'xcore1000' && assembly) {
    return <XCore1000AssemblyView assembly={assembly} />;
  }

  // tiny-gpu mode: show 16-bit binary
  if (instructions.length === 0) {
    return (
      <div style={{ padding: '16px', color: '#666', fontSize: '13px' }}>
        No instructions generated yet. Write a kernel above to compile.
      </div>
    );
  }

  return (
    <div style={{ overflow: 'auto', fontSize: '12px', fontFamily: 'monospace' }}>
      <table style={{ width: '100%', borderCollapse: 'collapse' }}>
        <thead>
          <tr style={{ background: '#111', color: '#888', textAlign: 'left' }}>
            <th style={{ padding: '6px 8px', width: '40px' }}>Addr</th>
            <th style={{ padding: '6px 8px', width: '60px' }}>Hex</th>
            <th style={{ padding: '6px 8px', width: '180px' }}>Binary</th>
            <th style={{ padding: '6px 8px' }}>Assembly</th>
          </tr>
        </thead>
        <tbody>
          {instructions.map((inst) => {
            const isHighlighted = inst.addr === highlightAddr;
            const binary = parseInt(inst.hex, 16) || 0;
            const opcode = (binary >> 12) & 0xf;
            const field1 = (binary >> 8) & 0xf;
            const field2 = (binary >> 4) & 0xf;
            const field3 = binary & 0xf;

            return (
              <tr
                key={inst.addr}
                style={{
                  background: isHighlighted ? '#2d3a2d' : 'transparent',
                  borderLeft: isHighlighted ? '3px solid #4ec9b0' : '3px solid transparent',
                  transition: 'background 0.15s',
                }}
              >
                <td style={{ padding: '4px 8px', color: '#666' }}>{inst.addr}</td>
                <td style={{ padding: '4px 8px', color: '#b5cea8' }}>{inst.hex}</td>
                <td style={{ padding: '4px 8px' }}>
                  <BinaryBits
                    opcode={opcode}
                    field1={field1}
                    field2={field2}
                    field3={field3}
                  />
                </td>
                <td style={{ padding: '4px 8px', color: '#e0e0e0' }}>{inst.asm}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

/** Render 16 bits with color coding per field */
function BinaryBits({
  opcode,
  field1,
  field2,
  field3,
}: {
  opcode: number;
  field1: number;
  field2: number;
  field3: number;
}) {
  const toBin4 = (n: number) => n.toString(2).padStart(4, '0');

  return (
    <span>
      <span style={{ color: '#e06c75' }} title="Opcode [15:12]">
        {toBin4(opcode)}
      </span>
      {' '}
      <span style={{ color: '#61afef' }} title="rd / NZP [11:8]">
        {toBin4(field1)}
      </span>
      {' '}
      <span style={{ color: '#98c379' }} title="rs [7:4]">
        {toBin4(field2)}
      </span>
      {' '}
      <span style={{ color: '#d19a66' }} title="rt / imm[3:0] [3:0]">
        {toBin4(field3)}
      </span>
    </span>
  );
}
